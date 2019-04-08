#define EIGEN_USE_GPU

#include "splines.h"
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
#define THREADS 256
#define BLOCKS 1024
#include <iostream>


const char *kernels = R"(
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

extern "C" {
__global__ void spline_grid_kernel_gpu(int N, int ndims, int n_neigh, int channels, float fill_value, const int *grid_dim_ptr, const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const int *periodic_ptr, const float *positions, const float *coefficients, float *out) {

	extern __shared__ int shared_info[];
	int *grid_dim = shared_info;
	int *strides = grid_dim + ndims;
	int *K = strides + ndims;
	int *dx = K + ndims;
	int *periodic = dx + ndims;

	// Let leader set shared memory
	if (threadIdx.x == 0) {
		for (int i = 0; i < ndims; i++) {
			grid_dim[i] = grid_dim_ptr[i];
			strides[i] = strides_ptr[i];
			K[i] = K_ptr[i];
			dx[i] = dx_ptr[i];
			periodic[i] = periodic_ptr[i];
		}
	}

	// All threads wait for leader
	__syncthreads();

	// Stride into shared memory
	int *idx = periodic + ndims + threadIdx.x;
	float *shift = (float*)(idx + ndims * blockDim.x);
	float *kernel_tmp = shift + ndims * blockDim.x;;

	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

		// Fetch the main index and shift
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			
			if (periodic[j]) {
				tmp = fmodf(tmp, 1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[blockDim.x*j] = modff(tmp*(grid_dim[j] + periodic[j] - 1) + 0.5, &tmp) - 0.5;
			idx[blockDim.x*j] = tmp;
		}

		// Reset channel sums
		for (int j = 0; j < channels; j++) {
			out[i*channels + j] = valid ? 0 : fill_value;
		}

		// Reduction loop over neighboring nodes
		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			int flat = 0;
			float Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[blockDim.x*k] + 1)) / 2 + (reduce % (K[k] + 1));

				if (periodic[k]) {
					flat += strides[k] * ((idx[blockDim.x*k] + offset + grid_dim[k]) % grid_dim[k]);
				}
				else {
					flat += strides[k] * fminf(fmaxf(idx[blockDim.x*k] + offset, 0), grid_dim[k] - 1);
				}
				Wij *= kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k]+periodic[k], float(dx[k]));
				reduce /= K[k] + 1;

			}
			// Accumulate contribution in each channel
			for (int k = 0; k < channels; k++) {
				if (valid) out[i*channels + k] += Wij * coefficients[channels*flat + k];
			}
		}
	}
}

__global__ void spline_grid_coefficient_gradient_kernel_gpu(int N, int ndims, int n_neigh, int channels, const int *grid_dim_ptr, const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const int *periodic_ptr, const float *positions, const float *grad, int *indices, float *values) {

	extern __shared__ int shared_info[];
	int *grid_dim = shared_info;
	int *strides = grid_dim + ndims;
	int *K = strides + ndims;
	int *dx = K + ndims;
	int *periodic = dx + ndims;

	// Let leader set shared memory
	if (threadIdx.x == 0) {
		for (int i = 0; i < ndims; i++) {
			grid_dim[i] = grid_dim_ptr[i];
			strides[i] = strides_ptr[i];
			K[i] = K_ptr[i];
			dx[i] = dx_ptr[i];
			periodic[i] = periodic_ptr[i];
		}
	}

	// All threads wait for leader
	__syncthreads();

	// Stride into shared memory
	int *idx = periodic + ndims + threadIdx.x;
	float *shift = (float*)(idx + ndims * blockDim.x);
	float *kernel_tmp = shift + ndims * blockDim.x;

	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

		// Fetch the main index and shift
		bool valid = true;
		for (int j = 0; j < ndims; j++) {
			float tmp = positions[i*ndims + j];
			
			if (periodic[j]) {
				tmp = fmodf(tmp, 1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[blockDim.x*j] = modff(tmp*(grid_dim[j] + periodic[j] - 1) + 0.5, &tmp) - 0.5;
			idx[blockDim.x*j] = tmp;
		}

		// Reduction loop over neighboring nodes
		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			float Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[blockDim.x*k] + 1)) / 2 + (reduce % (K[k] + 1));

				for (int l = 0; l < channels; l++) {
					if (periodic[k]) {
						indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + l * (ndims + 1) + k] = valid ? ((idx[blockDim.x*k] + offset + grid_dim[k]) % grid_dim[k]) : 0;
					}
					else {
						indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + l * (ndims + 1) + k] = valid ? fminf(fmaxf(idx[blockDim.x*k] + offset, 0), grid_dim[k] - 1) : 0;
					}
				}

				Wij *= kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k], float(dx[k]));
				reduce /= K[k] + 1;
			}
			for (int k = 0; k < channels; k++) {
				indices[i*n_neigh*channels*(ndims + 1) + j * channels*(ndims + 1) + k * (ndims + 1) + ndims] = k;
				values[i*n_neigh*channels + j * channels + k] = Wij * grad[i*channels + k];
			}
		}
	}
}

__global__ void spline_grid_position_gradient_kernel_gpu(int N, int ndims, int n_neigh, int channels, const int *grid_dim_ptr, const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const int *periodic_ptr, const float *positions, const float *coefficients, const float *grad, float *result) {

	extern __shared__ int shared_info[];
	int *grid_dim = shared_info;
	int *strides = grid_dim + ndims;
	int *K = strides + ndims;
	int *dx = K + ndims;
	int *periodic = dx + ndims;

	// Let leader set shared memory
	if (threadIdx.x == 0) {
		for (int i = 0; i < ndims; i++) {
			grid_dim[i] = grid_dim_ptr[i];
			strides[i] = strides_ptr[i];
			K[i] = K_ptr[i];
			dx[i] = dx_ptr[i];
			periodic[i] = periodic_ptr[i];
		}
	}

	// All threads wait for leader
	__syncthreads();

	// Stride into shared memory
	int *idx = periodic + ndims + threadIdx.x;
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
			
			if (periodic[j]) {
				tmp = fmodf(tmp, 1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[blockDim.x*j] = modff(tmp*(grid_dim[j] + periodic[j] - 1) + 0.5, &tmp) - 0.5;
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
				if(periodic[k]) {
					flat += strides[k] * ((idx[blockDim.x*k] + offset + grid_dim[k]) % grid_dim[k]);
				}
				else {
					flat += strides[k] * fminf(fmaxf(idx[blockDim.x*k] + offset, 0), grid_dim[k] - 1);
				}
				Wijs[blockDim.x*k] = kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k]+periodic[k], float(dx[k]));
				dWijs[blockDim.x*k] = kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k] + 1, kernel_tmp)*powf(grid_dim[k]+periodic[k], float((dx[k] + 1)));
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

} // End extern "C"

)";

static CUmodule module;
static CUfunction kernel;
static CUfunction kernel_coefficient_gradient;
static CUfunction kernel_position_gradient;

static bool compiled = false;

void compile() {
	if (compiled) {
		return;
	}
	cuInit(0);
	nvrtcProgram prog;
	nvrtcCreateProgram(&prog,
		kernels,
		"kernels.cu",
		0,
		NULL,
		NULL);

	const char **opts;
	nvrtcCompileProgram(prog,
		0,
		opts);

	size_t logSize;
	nvrtcGetProgramLogSize(prog, &logSize);
	char *log = new char[logSize];
	nvrtcGetProgramLog(prog, log);
	
	std::cout << log << std::endl;

	size_t ptxSize;
	nvrtcGetPTXSize(prog, &ptxSize);
	char *ptx = new char[ptxSize];
	nvrtcGetPTX(prog, ptx);

	nvrtcDestroyProgram(&prog);

	cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	cuModuleGetFunction(&kernel, module, "spline_grid_kernel_gpu");
	cuModuleGetFunction(&kernel_coefficient_gradient, module, "spline_grid_coefficient_gradient_kernel_gpu");
	cuModuleGetFunction(&kernel_position_gradient, module, "spline_grid_position_gradient_kernel_gpu");
	compiled = true;
}




template<typename T>
struct SplineGridFunctor<GPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *coefficients, float *out) {
		compile();

		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;

		int max_order = grid.maxorder();
		float fill_value = grid.fill_value;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		std::vector<int> periodic = grid.periodic;

		int *grid_dim_ptr, *strides_ptr, *K_ptr, *dx_ptr, *periodic_ptr;

		cudaMalloc(&grid_dim_ptr, ndims * sizeof(int));
		cudaMalloc(&strides_ptr, ndims * sizeof(int));
		cudaMalloc(&K_ptr, ndims * sizeof(int));
		cudaMalloc(&dx_ptr, ndims * sizeof(int));
		cudaMalloc(&periodic_ptr, ndims * sizeof(int));

		cudaMemcpy(grid_dim_ptr, grid_dim.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(strides_ptr, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(K_ptr, K.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dx_ptr, dx.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(periodic_ptr, periodic.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);

		// Compute shared memory size
		int shared_size = 5 * ndims * sizeof(int);
		shared_size += ndims * THREADS * sizeof(int);
		shared_size += ndims * THREADS * sizeof(float);
		shared_size += (max_order + 1) * THREADS * sizeof(float);
		
		void *args[] = { 
			&N, 
			&ndims, 
			&n_neigh, 
			&channels, 
			&fill_value, 
			&grid_dim_ptr, 
			&strides_ptr, 
			&K_ptr, 
			&dx_ptr, 
			&periodic_ptr, 
			&positions, 
			&coefficients, 
			&out};
		cuLaunchKernel(kernel,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			shared_size, NULL,
			args,
			0);

		// Free resources
		cudaFree(grid_dim_ptr);
		cudaFree(strides_ptr);
		cudaFree(K_ptr);
		cudaFree(dx_ptr);
		cudaFree(periodic_ptr);
	}

};


template struct SplineGridFunctor<GPU, float>;



template<typename T>
struct SplineGridCoefficientGradientFunctor<GPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *grad, int *indices, float *values) {
		compile();
		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		int max_order = grid.maxorder();
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		std::vector<int> periodic = grid.periodic;

		int *grid_dim_ptr, *strides_ptr, *K_ptr, *dx_ptr, *periodic_ptr;

		cudaMalloc(&grid_dim_ptr, ndims * sizeof(int));
		cudaMalloc(&strides_ptr, ndims * sizeof(int));
		cudaMalloc(&K_ptr, ndims * sizeof(int));
		cudaMalloc(&dx_ptr, ndims * sizeof(int));
		cudaMalloc(&periodic_ptr, ndims * sizeof(int));

		cudaMemcpy(grid_dim_ptr, grid_dim.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(strides_ptr, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(K_ptr, K.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dx_ptr, dx.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(periodic_ptr, periodic.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);

		// Compute shared memory size
		int shared_size = 5 * ndims * sizeof(int);
		shared_size += ndims * THREADS * sizeof(int);
		shared_size += ndims * THREADS * sizeof(float);
		shared_size += (max_order + 1) * THREADS * sizeof(float);


		void *args[] = {
			&N,
			&ndims,
			&n_neigh,
			&channels,
			&grid_dim_ptr,
			&strides_ptr,
			&K_ptr,
			&dx_ptr,
			&periodic_ptr,
			&positions,
			&grad,
			&indices,
			&values
		};
		cuLaunchKernel(kernel_coefficient_gradient,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			shared_size, NULL,
			args,
			0);

		// Free resources
		cudaFree(grid_dim_ptr);
		cudaFree(strides_ptr);
		cudaFree(K_ptr);
		cudaFree(dx_ptr);
		cudaFree(periodic_ptr);
	}

};


template struct SplineGridCoefficientGradientFunctor<GPU, float>;



template<typename T>
struct SplineGridPositionGradientFunctor<GPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *coefficients, const float *grad, float *result) {
		
		compile();
		int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
		int max_order = grid.maxorder();
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		std::vector<int> periodic = grid.periodic;

		int *grid_dim_ptr, *strides_ptr, *K_ptr, *dx_ptr, *periodic_ptr;

		cudaMalloc(&grid_dim_ptr, ndims * sizeof(int));
		cudaMalloc(&strides_ptr, ndims * sizeof(int));
		cudaMalloc(&K_ptr, ndims * sizeof(int));
		cudaMalloc(&dx_ptr, ndims * sizeof(int));
		cudaMalloc(&periodic_ptr, ndims * sizeof(int));

		cudaMemcpy(grid_dim_ptr, grid_dim.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(strides_ptr, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(K_ptr, K.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dx_ptr, dx.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(periodic_ptr, periodic.data(), ndims * sizeof(int), cudaMemcpyHostToDevice);

		// Compute shared memory size
		int shared_size = 5 * ndims * sizeof(int);
		shared_size += ndims * THREADS * sizeof(int);
		shared_size += ndims * THREADS * sizeof(float);
		shared_size += (max_order + 1) * THREADS * sizeof(float);
		shared_size += (ndims)* THREADS * sizeof(float);
		shared_size += (ndims)* THREADS * sizeof(float);
		shared_size += (ndims)* THREADS * sizeof(float);


		void *args[] = {
			&N, 
			&ndims, 
			&n_neigh, 
			&channels, 
			&grid_dim_ptr, 
			&strides_ptr, 
			&K_ptr, 
			&dx_ptr, 
			&periodic_ptr, 
			&positions, 
			&coefficients, 
			&grad, 
			&result
		};
		cuLaunchKernel(kernel_position_gradient,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			shared_size, NULL,
			args,
			0);

		// Free resources
		cudaFree(grid_dim_ptr);
		cudaFree(strides_ptr);
		cudaFree(K_ptr);
		cudaFree(dx_ptr);
		cudaFree(periodic_ptr);
	}

};


template struct SplineGridPositionGradientFunctor<GPU, float>;
