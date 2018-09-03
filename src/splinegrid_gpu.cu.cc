#include "splines.h"

#define THREADS 64

//GPU specialization of actual computation.
__device__
float kernel_gpu(float x, int p, int dx, float *tmp) {
	if (dx > p) {
		return 0;
	}
	x += (p + 1) / 2.;
	int k = x;
	for (int i = 0; i < p + 1; i++) {
		tmp[blockDim.x*i] = k == i;
	}

	for (int i = 0; i < p; i++) {
		for (int j = 0; j < p - i; j++) {
			tmp[blockDim.x*j] = i < p - dx ? (x - j) / (i + 1)*tmp[blockDim.x*j] + (i + 2 + j - x) / (i + 1)*tmp[blockDim.x*(j + 1)] : tmp[blockDim.x*j] - tmp[blockDim.x*(j + 1)];
		}
	}
	return tmp[0];
}

__global__ void spline_grid_kernel_gpu(int N, int ndims, int n_neigh, int channels, float fill_value, bool normalized, const int *grid_dim_ptr, const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const float *positions, const float *coefficients, float *out) {

	extern __shared__ int shared_info[];
	int *grid_dim = shared_info;
	int *strides = grid_dim + ndims;
	int *K = strides + ndims;
	int *dx = K + ndims;

	// Let leader set shared memory
	if (threadIdx.x == 0) {
		for (int i = 0; i < ndims; i++) {
			grid_dim[i] = grid_dim_ptr[i];
			strides[i] = strides_ptr[i];
			K[i] = K_ptr[i];
			dx[i] = dx_ptr[i];
		}
	}

	// All threads wait for leader
	__syncthreads();

	// Stride into shared memory
	int *idx = dx + ndims + threadIdx.x;
	float *shift = (float*)(idx + ndims * blockDim.x);
	float *channel_sum = shift + ndims * blockDim.x;
	float *kernel_tmp = channel_sum + channels * blockDim.x;

	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

		// Fetch the main index and shift
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			if (!normalized) {
				tmp /= grid_dim[j];
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[blockDim.x*j] = modff(tmp*(grid_dim[j] - 1) + 0.5, &tmp) - 0.5;
			idx[blockDim.x*j] = tmp;
		}

		// Reset channel sums
		for (int j = 0; j < channels; j++) {
			channel_sum[blockDim.x*j] = 0;
		}

		// Reduction loop over neighboring nodes
		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			int flat = 0;
			float Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[blockDim.x*k] + 1)) / 2 + (reduce % (K[k] + 1));

				flat += strides[k] * fminf(fmaxf(idx[blockDim.x*k] + offset, 0), grid_dim[k] - 1);
				Wij *= kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k], float(normalized*dx[k]));
				reduce /= K[k] + 1;

			}
			// Accumulate contribution in each channel
			for (int k = 0; k < channels; k++) {
				channel_sum[blockDim.x*k] += Wij * (valid ? coefficients[channels*flat + k] : 0);
			}
		}
		// Write channel sum to global memory
		for (int j = 0; j < channels; j++) {
			out[i*channels + j] = valid ? channel_sum[blockDim.x*j] : fill_value;
		}
	}
}

template<typename T>
struct SplineGridFunctor<GPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *coefficients, float *out) {


		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		int max_order = grid.maxorder();
		float fill_value = grid.fill_value;
		bool normalized = grid.normalized;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;

		int *grid_dim_ptr, *strides_ptr, *K_ptr, *dx_ptr;

		cudaMalloc(&grid_dim_ptr, ndims * sizeof(int));
		cudaMalloc(&strides_ptr, ndims * sizeof(int));
		cudaMalloc(&K_ptr, ndims * sizeof(int));
		cudaMalloc(&dx_ptr, ndims * sizeof(int));

		cudaMemcpy(grid_dim_ptr, grid_dim.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(strides_ptr, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(K_ptr, K.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dx_ptr, dx.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);

		// Compute shared memory size
		int shared_size = 4 * ndims * sizeof(int);
		shared_size += ndims * THREADS * sizeof(int);
		shared_size += ndims * THREADS * sizeof(float);
		shared_size += channels * THREADS * sizeof(float);
		shared_size += (max_order + 1) * THREADS * sizeof(float);

		// Enqueue kernel
		spline_grid_kernel_gpu<<<80, THREADS, shared_size>>> (N, ndims, n_neigh, channels, fill_value, normalized, grid_dim_ptr, strides_ptr, K_ptr, dx_ptr, positions, coefficients, out);


		// Free resources
		cudaFree(grid_dim_ptr);
		cudaFree(strides_ptr);
		cudaFree(K_ptr);
		cudaFree(dx_ptr);
	}

};


template struct SplineGridFunctor<GPU, float>;

//GPU specialization of actual computation.

__global__ void spline_grid_gradient_kernel_gpu(int N, int ndims, int n_neigh, int channels, bool normalized, const int *grid_dim_ptr, const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const float *positions, const float *grad, int *indices, float *values) {

	extern __shared__ int shared_info[];
	int *grid_dim = shared_info;
	int *strides = grid_dim + ndims;
	int *K = strides + ndims;
	int *dx = K + ndims;

	// Let leader set shared memory
	if (threadIdx.x == 0) {
		for (int i = 0; i < ndims; i++) {
			grid_dim[i] = grid_dim_ptr[i];
			strides[i] = strides_ptr[i];
			K[i] = K_ptr[i];
			dx[i] = dx_ptr[i];
		}
	}

	// All threads wait for leader
	__syncthreads();

	// Stride into shared memory
	int *idx = dx + ndims + threadIdx.x;
	float *shift = (float*)(idx + ndims * blockDim.x);
	float *kernel_tmp = shift + ndims * blockDim.x;

	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

		// Fetch the main index and shift
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			if (!normalized) {
				tmp /= grid_dim[j];
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[blockDim.x*j] = modff(tmp*(grid_dim[j] - 1) + 0.5, &tmp) - 0.5;
			idx[blockDim.x*j] = tmp;
		}

		// Reduction loop over neighboring nodes
		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			float Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[blockDim.x*k] + 1)) / 2 + (reduce % (K[k] + 1));

				for (int l = 0; l < channels; l++) {
					indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + l * (ndims + 1) + k] = fminf(fmaxf(idx[blockDim.x*k] + offset, 0), grid_dim[k] - 1);
				}

				Wij *= kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k], float(normalized*dx[k]));
				reduce /= K[k] + 1;
			}
			for (int k = 0; k < channels; k++) {
				indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + k * (ndims + 1) + ndims] = k;
				values[i*n_neigh*channels + j * channels + k] = Wij * grad[i*channels + k];
			}
		}
	}
}

template<typename T>
struct SplineGridGradientFunctor<GPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *grad, int *indices, float *values) {

		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		int max_order = grid.maxorder();
		bool normalized = grid.normalized;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;

		int *grid_dim_ptr, *strides_ptr, *K_ptr, *dx_ptr;

		cudaMalloc(&grid_dim_ptr, ndims * sizeof(int));
		cudaMalloc(&strides_ptr, ndims * sizeof(int));
		cudaMalloc(&K_ptr, ndims * sizeof(int));
		cudaMalloc(&dx_ptr, ndims * sizeof(int));

		cudaMemcpy(grid_dim_ptr, grid_dim.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(strides_ptr, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(K_ptr, K.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dx_ptr, dx.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);

		// Compute shared memory size
		int shared_size = 4 * ndims * sizeof(int);
		shared_size += ndims * THREADS * sizeof(int);
		shared_size += ndims * THREADS * sizeof(float);
		shared_size += (max_order + 1) * THREADS * sizeof(float);

		// Enqueue kernel
		spline_grid_gradient_kernel_gpu<<<80, THREADS, shared_size>>> (N, ndims, n_neigh, channels, normalized, grid_dim_ptr, strides_ptr, K_ptr, dx_ptr, positions, grad, indices, values);

		// Free resources
		cudaFree(grid_dim_ptr);
		cudaFree(strides_ptr);
		cudaFree(K_ptr);
		cudaFree(dx_ptr);
	}

};


template struct SplineGridGradientFunctor<GPU, float>;




__global__ void spline_grid_gradient_kernel_gpu(int N, int ndims, int n_neigh, int channels, bool normalized, const int *grid_dim_ptr, const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const float *positions, const float *coefficients, const float *grad, float *result) {

	extern __shared__ int shared_info[];
	int *grid_dim = shared_info;
	int *strides = grid_dim + ndims;
	int *K = strides + ndims;
	int *dx = K + ndims;

	// Let leader set shared memory
	if (threadIdx.x == 0) {
		for (int i = 0; i < ndims; i++) {
			grid_dim[i] = grid_dim_ptr[i];
			strides[i] = strides_ptr[i];
			K[i] = K_ptr[i];
			dx[i] = dx_ptr[i];
		}
	}

	// All threads wait for leader
	__syncthreads();

	// Stride into shared memory
	int *idx = dx + ndims + threadIdx.x;
	float *shift = (float*)(idx + ndims * blockDim.x);
	float *directional_diff = shift + ndims * blockDim.x;
	float *Wijs = directional_diff + ndims * blockDim.x;
	float *dWijs = Wijs + ndims * blockDim.x;

	float *kernel_tmp = dWijs + ndims * blockDim.x;



	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

		// Fetch the main index and shift
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			if (!normalized) {
				tmp /= grid_dim[j];
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[blockDim.x*j] = modff(tmp*(grid_dim[j] - 1) + 0.5, &tmp) - 0.5;
			idx[blockDim.x*j] = tmp;
		}

		for (int j = 0; j < ndims; j++) {
			directional_diff[blockDim.x*j] = 0;
		}

		// Reduction loop over neighboring nodes
		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			int flat = 0;

			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[blockDim.x*k] + 1)) / 2 + (reduce % (K[k] + 1));

				flat += strides[k] * fminf(fmaxf(idx[blockDim.x*k] + offset, 0), grid_dim[k] - 1);
				Wijs[blockDim.x*k] = kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k], float(normalized*dx[k]));
				dWijs[blockDim.x*k] = kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k] + 1, kernel_tmp)*powf(grid_dim[k], float(normalized*(dx[k] + 1)));
				reduce /= K[k] + 1;
			}

			float channel_sum = 0;
			for (int k = 0; k < channels; k++) {
				channel_sum += valid ? coefficients[flat*channels + k] * grad[i*channels + k] : 0;
			}

			for (int k = 0; k < ndims; k++) {
				float coeff = 1;
				for (int l = 0; l < ndims; l++) {
					coeff *= l == k ? dWijs[blockDim.x*l] : Wijs[blockDim.x*l];
				}
				directional_diff[blockDim.x*k] += coeff * channel_sum;
			}
		}
		for (int j = 0; j < ndims; j++) {
			result[i*ndims + j] = valid ? directional_diff[blockDim.x*j] : 0;
		}
	}
}



template<typename T>
struct SplineGridPositionGradientFunctor<GPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *coefficients, const float *grad, float *result) {

		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		int max_order = grid.maxorder();
		bool normalized = grid.normalized;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;

		int *grid_dim_ptr, *strides_ptr, *K_ptr, *dx_ptr;

		cudaMalloc(&grid_dim_ptr, ndims * sizeof(int));
		cudaMalloc(&strides_ptr, ndims * sizeof(int));
		cudaMalloc(&K_ptr, ndims * sizeof(int));
		cudaMalloc(&dx_ptr, ndims * sizeof(int));

		cudaMemcpy(grid_dim_ptr, grid_dim.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(strides_ptr, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(K_ptr, K.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dx_ptr, dx.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);

		// Compute shared memory size
		int shared_size = 4 * ndims * sizeof(int);
		shared_size += ndims * THREADS * sizeof(int);
		shared_size += ndims * THREADS * sizeof(float);
		shared_size += (max_order + 1) * THREADS * sizeof(float);
		shared_size += (ndims)* THREADS * sizeof(float);
		shared_size += (ndims)* THREADS * sizeof(float);
		shared_size += (ndims)* THREADS * sizeof(float);
		// Enqueue kernel
		spline_grid_gradient_kernel_gpu << <80, THREADS, shared_size >> > (N, ndims, n_neigh, channels, normalized, grid_dim_ptr, strides_ptr, K_ptr, dx_ptr, positions, coefficients, grad, result);

		// Free resources
		cudaFree(grid_dim_ptr);
		cudaFree(strides_ptr);
		cudaFree(K_ptr);
		cudaFree(dx_ptr);
	}

};


template struct SplineGridPositionGradientFunctor<GPU, float>;