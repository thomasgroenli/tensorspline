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


void bspline_kernel_cpu(int start, int end, int order, int dx, const float *x, float *bsx) {
	float *kernel_tmp = new float[order + 1];
	for (int i = start; i < end; ++i) {
		bsx[i] = kernel_cpu(x[i], order, dx, kernel_tmp);
	}
}

template<typename T>
struct BSplineFunctor<CPU, T> {
	void operator()(OpKernelContext *context,  int N, int order, int dx, const float *x, float *bsx) {

#ifdef USE_MULTITHREAD
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(pool->NumThreads(), pool, N, 1024, [&](int start, int end) {
			bspline_kernel_cpu(start, end, order, dx, x, bsx);	
		});
#else
		bspline_kernel_cpu(0, N, order, dx, x, bsx);
#endif
	}
};

template struct BSplineFunctor<CPU, float>;

void spline_grid_kernel_cpu(int start, int end, int ndims, int n_neigh, int channels, float fill_value, const int *grid_dim, const int *strides, const int *K, const int *dx, const int *periodic, const float *positions, const float *coefficients, float *out) {
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
			
			if (periodic[j]) {
				tmp = fmod(tmp,1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[j] = modff(tmp*(grid_dim[j] + periodic[j] - 1) + 0.5, &tmp) - 0.5;
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
				
				if (periodic[k]) {
					flat += strides[k] * ((idx[k] + offset + grid_dim[k]) % grid_dim[k]);
				}
				else {
					int in_pos = idx[k] + offset;
					flat += strides[k] * (in_pos>=grid_dim[k]?2*(grid_dim[k]-1)-in_pos:fabs(in_pos)); 
				}
				Wij *= kernel_cpu(shift[k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k]-1+periodic[k], float(dx[k]));
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
}

template<typename T>
struct SplineGridFunctor<CPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *coefficients, float *out) {

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
	}
};

template struct SplineGridFunctor<CPU, float>;

void spline_grid_coefficient_gradient_kernel_cpu(int start, int end, int ndims, int n_neigh, int channels, const int *grid_dim, const int *strides, const int *K, const int *dx, const int *periodic, const float *positions, const float *grad, int *indices, float *values) {
	int *idx = new int[ndims];
	float *shift = new float[ndims];

	int max_order = 0;
	for (int i = 0; i < ndims; i++) {
		max_order = K[i] > max_order ? K[i] : max_order;
	}
	float *kernel_tmp = new float[max_order + 1];


	for (int i = start; i < end; i++) {
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			
			if (periodic[j]) {
				tmp = fmod(tmp, 1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[j] = modff(tmp*(grid_dim[j] + periodic[j] - 1) + 0.5, &tmp) - 0.5;
			idx[j] = tmp;
		}

		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			float Wij = 1;

			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[k] + 1)) / 2 + (reduce % (K[k] + 1));

				int out_pos;
				if (periodic[k]) {
					out_pos = (idx[k] + offset + grid_dim[k]) % grid_dim[k];
				}
				else {
					int in_pos = idx[k] + offset;
					out_pos = in_pos>=grid_dim[k]?2*(grid_dim[k]-1)-in_pos:abs(in_pos); 
				}
				for (int l = 0; l < channels; l++) {
					indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + l * (ndims + 1) + k] = valid ? out_pos : 0;
				}
				Wij *= kernel_cpu(shift[k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k]-1+periodic[k], float(dx[k]));
				reduce /= K[k] + 1;
			}
			for (int k = 0; k < channels; k++) {
				indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + k * (ndims + 1) + ndims] = valid ? k : 0;
				values[i*n_neigh*channels + j * channels + k] = valid ? Wij * grad[i*channels + k] : 0;
			}
		}
	}
	delete[] idx;
	delete[] shift;
	delete[] kernel_tmp;
}

template<typename T>
struct SplineGridCoefficientGradientFunctor<CPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float* grad, int *indices, float *values) {

		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		std::vector<int> periodic = grid.periodic;

/*#ifdef USE_MULTITHREAD
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(pool->NumThreads(), pool, N, 1024, [&](int start, int end) {
			spline_grid_coefficient_gradient_kernel_cpu(start, end, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), dx.data(), periodic.data(), positions, grad, indices, values);
		});
#else*/
		spline_grid_coefficient_gradient_kernel_cpu(0, N, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), dx.data(), periodic.data(), positions, grad, indices, values);
//#endif
	}
};


template struct SplineGridCoefficientGradientFunctor<CPU, float>;

void spline_grid_position_gradient_kernel_cpu(int start, int end, int ndims, int n_neigh, int channels, const int *grid_dim, const int *strides, const int *K, const int *dx, const int *periodic, const float *positions, const float *coefficients, const float *grad, float *result) {
	int *idx = new int[ndims];
	float *shift = new float[ndims];

	int max_order = 0;
	for (int i = 0; i < ndims; i++) {
		max_order = K[i] > max_order ? K[i] : max_order;
	}
	float *kernel_tmp = new float[max_order + 1];
	
	float *directional_diff = new float[ndims];
	float *Wijs = new float[ndims];
	float *dWijs = new float[ndims];

	for (int i = start; i < end; i++) {
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			
			if (periodic[j]) {
				tmp = fmod(tmp, 1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[j] = modff(tmp*(grid_dim[j] + periodic[j] - 1) + 0.5, &tmp) - 0.5;
			idx[j] = tmp;
		}

		for (int j = 0; j < ndims; j++) {
			directional_diff[j] = 0;
		}

		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			int flat = 0;

			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[k] + 1)) / 2 + (reduce % (K[k] + 1));
				if (periodic[k]) {
					flat += strides[k] * ((idx[k] + offset + grid_dim[k]) % grid_dim[k]);
				}
				else {
					int in_pos = idx[k] + offset;
					flat += strides[k] * (in_pos>=grid_dim[k]?2*(grid_dim[k]-1)-in_pos:fabs(in_pos)); 
				}
				Wijs[k] = kernel_cpu(shift[k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k]-1+periodic[k], float(dx[k]));
				dWijs[k] = kernel_cpu(shift[k] - offset, K[k], dx[k]+1, kernel_tmp)*powf(grid_dim[k]-1+periodic[k], float(dx[k]+1));
				reduce /= K[k] + 1;
			}

			float channel_sum = 0;
			for (int k = 0; k < channels; k++) {
				channel_sum += valid?coefficients[flat*channels + k]*grad[i*channels+k]:0;
			}

			for (int k = 0; k < ndims; k++) {
				float coeff = 1;
				for (int l = 0; l < ndims; l++) {
					coeff *= l == k ? dWijs[l] : Wijs[l];
				}
				directional_diff[k] += coeff*channel_sum;
			}
		}
		for (int j = 0; j < ndims; j++) {
			result[i*ndims + j] = valid?directional_diff[j]:0;
		}
	}
	delete[] idx;
	delete[] shift;
	delete[] kernel_tmp;
	delete[] Wijs;
	delete[] dWijs;
	delete[] directional_diff;
}

template<typename T>
struct SplineGridPositionGradientFunctor<CPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *coefficients, const float* grad, float *result) {

		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		std::vector<int> periodic = grid.periodic;

#ifdef USE_MULTITHREAD
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(pool->NumThreads(), pool, N, 1024, [&](int start, int end) {
			spline_grid_position_gradient_kernel_cpu(start, end, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), dx.data(), periodic.data(), positions, coefficients, grad, result);
		});
#else
		spline_grid_position_gradient_kernel_cpu(0, N, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), dx.data(), periodic.data(), positions, coefficients, grad, result);
#endif

	}
};



template struct SplineGridPositionGradientFunctor<CPU, float>;