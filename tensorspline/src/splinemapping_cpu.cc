#include "splines.h"


void spline_mapping_kernel_cpu(int start, int end, int ndims, int n_neigh, int channels, const int *grid_dim, const int *strides, const int *K, const int *dx, const int *periodic, const float *positions, const float *values, const float *weights, float *grid, float *density, lock *locks) {
int *idx = new int[ndims];
	float *shift = new float[ndims];
	int max_order = 0;
	for (int i = 0; i < ndims; i++) {
		max_order = K[i] > max_order ? K[i] : max_order;
	}
	float *kernel_tmp = new float[max_order + 1];
    bool lock_var = false;
	for (int i = start; i < end; ++i) {
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			
			if (periodic[j]) {
				tmp = fmod(tmp,1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[j] = modff(tmp*(grid_dim[j] + periodic[j] - 1) + 0.5, &tmp) - 0.5;
			idx[j] = tmp;
		}
		
		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			int flat = 0;
			float Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[k] + 1)) / 2 + (reduce % (K[k] + 1));

				if (periodic[k]) {
					flat += strides[k] * ((idx[k] + offset + grid_dim[k]) % grid_dim[k]);
				}
				else {
					flat += strides[k] * fmin(fmax(idx[k] + offset, 0), grid_dim[k] - 1);
				}
				Wij *= kernel_cpu(shift[k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k]-1+periodic[k], float(dx[k]));
				reduce /= K[k] + 1;
			}
            if(valid) {
                while(!locks[flat].compare_exchange_strong(lock_var, true)) {lock_var=false;}

                for (int k = 0; k < channels; k++) {
                    grid[flat*channels + k] += weights[i*channels + k] * Wij * values[channels*i + k];
					density[flat*channels + k] += weights[i*channels + k] * Wij;
                }

                locks[flat].store(false);
            }
			
		}
	}
	delete[] idx;
	delete[] shift;
	delete[] kernel_tmp;
}



template<typename T>
struct SplineMappingFunctor<CPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *values, const float *weights, float *output_grid) {
		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		std::vector<int> periodic = grid.periodic;

        lock *locks = new lock[grid.num_points()];
        float *density = new float[channels*grid.num_points()];
        for(int i=0; i<grid.num_points(); i++) {
            locks[i] = false;
            
			for(int j=0; j<channels; j++) {
				output_grid[i*channels+j] = 0;
				density[i*channels + j] = 0;
			}
        }

#ifdef USE_MULTITHREAD
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(pool->NumThreads(), pool, N, 1024, [&](int start, int end) {
			spline_mapping_kernel_cpu(start, end, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), dx.data(), periodic.data(), positions, values, weights, output_grid, density, locks);
		});
#else
		spline_mapping_kernel_cpu(0, N, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), dx.data(), periodic.data(), positions, values, weights, output_grid, density, locks);
#endif

        for(int i=0; i<grid.num_points(); i++) {
			for(int j=0; j<channels; j++) {
				output_grid[channels*i+j] /= density[channels*i+j];
			}
        }

        delete[] locks;
        delete[] density;
	}
};


template struct SplineMappingFunctor<CPU, float>;