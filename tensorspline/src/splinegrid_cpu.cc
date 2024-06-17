#include "splines.h"

//CPU specialization of actual computation.
float kernel_cpu(float x, int p, int dx, float *tmp) {
	if (dx > p) {
		return 0;
	}
	x += (p + 1) / 2.;
	int k = floor(x);
	for (int i = 0; i < p + 1; i++) {
		tmp[i] = k == i;
	}

	for (int i = 0; i < p; i++) {
		for (int j = 0; j < p - i; j++) {
			tmp[j] = i < p - dx ? (x - j) / (i + 1)*tmp[j] + (i + 2 + j - x) / (i + 1)*tmp[j + 1] : tmp[j] - tmp[j + 1];
		}
	}
	return tmp[0];
}


Status bspline_kernel_cpu(int start, int end, int order, int dx, const float *x, float *bsx) {
	float *kernel_tmp = new float[order + 1];
	for (int i = start; i < end; ++i) {
		bsx[i] = kernel_cpu(x[i], order, dx, kernel_tmp);
	}
	return absl::OkStatus();
}

template<typename T>
struct BSplineFunctor<CPU, T> {
	Status operator()(OpKernelContext *context,  int N, int order, int dx, const float *x, float *bsx) {

#ifdef USE_MULTITHREAD
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(pool->NumThreads(), pool, N, 1024, [&](int start, int end) {
			bspline_kernel_cpu(start, end, order, dx, x, bsx);	
		});
#else
		bspline_kernel_cpu(0, N, order, dx, x, bsx);
#endif
	return absl::OkStatus();
	}
};

template struct BSplineFunctor<CPU, float>;

Status spline_grid_kernel_cpu(int start, int end, int ndims, int n_neigh, int channels, float fill_value, const int *grid_dim, const int *strides, const int *K, const int *dx, const int *periodic, const float *positions, const float *coefficients, float *out) {
	int *idx = new int[ndims];
	float *shift = new float[ndims];
	int max_order = 0;
	for (int i = 0; i < ndims; i++) {
		max_order = K[i] > max_order ? K[i] : max_order;
	}
	float *kernel_tmp = new float[max_order + 1];

	for (int i = start; i < end; ++i) {
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			
			if (periodic[j]==1) {
				tmp = fmod(tmp,1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[j] = modff(tmp*(grid_dim[j] + (periodic[j]==1) - 1) + 0.5, &tmp) - 0.5;
			idx[j] = tmp;
		}
		for (int j = 0; j < channels; j++) {
			out[i*channels + j] = valid ? 0 : fill_value;
		}
		
		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			int flat = 0;
			float Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[k] + 1)) / 2 + (reduce % (K[k] + 1));
				
				int in_span = grid_dim[k];
				int in_pos = idx[k]+offset;

				if(periodic[k]==1) {
					flat += strides[k] * positive_modulo(in_pos, in_span);
				} else if(periodic[k]==-1) {
					int reflect = positive_modulo(in_pos, 2*(in_span-1));
					flat += strides[k] * (reflect<in_span?reflect:2*(in_span-1)-reflect); 
				} else {
					flat += strides[k] * fmin(fmax(in_pos, 0), in_span-1);
				}

				Wij *= kernel_cpu(shift[k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k]-1+(periodic[k]==1), float(dx[k]));
				reduce /= K[k] + 1;
			}
			for (int k = 0; k < channels; k++) {
				if (valid) out[i*channels + k] += Wij * coefficients[channels*flat + k];
			}
		}
	}
	delete[] idx;
	delete[] shift;
	delete[] kernel_tmp;

	return absl::OkStatus();
}

template<typename T>
struct SplineGridFunctor<CPU, T> {
	Status operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *coefficients, float *out) {

		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		float fill_value = grid.fill_value;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		std::vector<int> periodic = grid.periodic;

#ifdef USE_MULTITHREAD
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(pool->NumThreads(), pool, N, 1024, [&](int start, int end) {
			spline_grid_kernel_cpu(start, end, ndims, n_neigh, channels, fill_value, grid_dim.data(), strides.data(), K.data(), dx.data(), periodic.data(), positions, coefficients, out);
		});
#else
		spline_grid_kernel_cpu(0, N, ndims, n_neigh, channels, fill_value, grid_dim.data(), strides.data(), K.data(), dx.data(), periodic.data(), positions, coefficients, out);
#endif

		return absl::OkStatus();
	}
};

template struct SplineGridFunctor<CPU, float>;