#include "splines.h"

//CPU specialization of actual computation.
float kernel_cpu(float x, int p, int dx, float *tmp) {
	if (dx > p) {
		return 0;
	}
	x += (p + 1) / 2.;
	int k = x;
	for (int i = 0; i < p + 1; i++) {
		tmp[i] = k==i;
	}

	for (int i = 0; i < p; i++) {
		for (int j = 0; j < p - i; j++) {
			tmp[j] = i < p - dx ? (x - j) / (i + 1)*tmp[j] + (i + 2 + j - x) / (i + 1)*tmp[j + 1] : tmp[j] - tmp[j + 1];
		}
	}
	return tmp[0];
}

void spline_grid_kernel_cpu(int start, int end, int ndims, int n_neigh, int channels, float fill_value, bool normalized, const int *grid_dim, const int *strides, const int *K, const int *dx, const float *positions, const float *coefficients, float *out) {
	int *idx = new int[ndims];
	float *shift = new float[ndims];
	float *channel_sum = new float[channels];
	int max_order = 0;
	for (int i = 0; i < ndims; i++) {
		max_order = K[i] > max_order ? K[i] : max_order;
	}
	float *kernel_tmp = new float[max_order + 1];

	float tmp;
	int reduce;
	int flat;
	float Wij;


	for (int i = start; i < end; ++i) {
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			tmp = positions[i*ndims + j];
			if (!normalized) {
				tmp /= grid_dim[j];
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[j] = modff(tmp*(grid_dim[j] - 1) + 0.5, &tmp) - 0.5;
			idx[j] = tmp;
		}
		for (int j = 0; j < channels; j++) {
			channel_sum[j] = 0;
		}

		for (int j = 0; j < n_neigh; j++) {
			reduce = j;
			flat = 0;
			Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[k] + 1)) / 2 + (reduce % (K[k] + 1));
				int current = idx[k] + offset;
				float x = shift[k] - offset;
				current = fmin(fmax(current, 0), grid_dim[k] - 1);

				flat += strides[k] * current;
				Wij *= kernel_cpu(x, K[k], dx[k], kernel_tmp);
				reduce /= K[k] + 1;
			}
			for (int k = 0; k < channels; k++) {
				channel_sum[k] += Wij * (valid ? coefficients[channels*flat + k] : 0);
			}
		}
		for (int j = 0; j < channels; j++) {
			out[i*channels + j] = valid ? channel_sum[j] : fill_value;
		}
	}
	delete[] idx, shift, channel_sum, kernel_tmp;
}

template<typename T>
struct SplineGridFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *coefficients, float *out) {
    
	int ndims = grid.ndims();
    int n_neigh = grid.neighbors();
    int channels = grid.channels;
	float fill_value = grid.fill_value;
	bool normalized = grid.normalized;
    std::vector<int> strides = grid.strides();
    std::vector<int> grid_dim = grid.dims;
    std::vector<int> K = grid.K;
    std::vector<int> dx = grid.dx;

	auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
	Shard(pool->NumThreads(), pool, N, 128, [&](int start, int end) {
		spline_grid_kernel_cpu(start, end, ndims, n_neigh, channels, fill_value, normalized, grid_dim.data(), strides.data(), K.data(), dx.data(), positions, coefficients, out);
	});
  }
     
};

template struct SplineGridFunctor<Eigen::ThreadPoolDevice, float>;

void spline_grid_gradient_kernel_cpu(int start, int end, int ndims, int n_neigh, int channels, bool normalized, const int *grid_dim, const int *strides, const int *K, const int *dx, const float *positions, const float *grad, int *indices, float *values) {
	int *idx = new int[ndims];
	float *shift = new float[ndims];

	int max_order = 0;
	for (int i = 0; i < ndims; i++) {
		max_order = K[i] > max_order ? K[i] : max_order;
	}
	float *kernel_tmp = new float[max_order + 1];

	float tmp;
	int reduce;
	float Wij;

	for (int i = start; i<end; i++) {
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			tmp = positions[i*ndims + j];
			if (!normalized) {
				tmp /= grid_dim[j];
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[j] = modff(tmp*(grid_dim[j] - 1) + 0.5, &tmp) - 0.5;
			idx[j] = tmp;
		}

		for (int j = 0; j<n_neigh; j++) {
			reduce = j;
			Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[k] + 1)) / 2 + (reduce % (K[k] + 1));
				int current = idx[k] + offset;
				float x = shift[k] - offset;
				current = fmin(fmax(current, 0), grid_dim[k] - 1);

				for (int l = 0; l<channels; l++) {
					indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + l * (ndims + 1) + k] = valid ? current : 0;
				}
				Wij *= kernel_cpu(x, K[k], dx[k], kernel_tmp);
				reduce /= K[k] + 1;
			}
			for (int k = 0; k<channels; k++) {
				indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + k * (ndims + 1) + ndims] = valid ? k: 0;
				values[i*n_neigh*channels + j * channels + k] = valid ? Wij * grad[i*channels + k] : 0;
			}
		}
	}
	delete[] idx, shift, kernel_tmp;
}

template<typename T>
struct SplineGridGradientFunctor<Eigen::ThreadPoolDevice, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float* grad, int *indices, float *values) {

		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		bool normalized = grid.normalized;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(1/*pool->NumThreads()*/, pool, N, 256, [&](int start, int end) {
			spline_grid_gradient_kernel_cpu(start, end, ndims, n_neigh, channels, normalized, grid_dim.data(), strides.data(), K.data(), dx.data(), positions, grad, indices, values);
		});
	}

};


template struct SplineGridGradientFunctor<Eigen::ThreadPoolDevice, float>;