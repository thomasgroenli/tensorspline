#define EIGEN_USE_GPU

#include "splines.h"
#include <iostream>


const char *kernels_map = R"(
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

__device__ int positive_modulo(int i, int n) {
	if(n==0) {
		return 0;
	}
    return (i % n + n) % n;
}

extern "C" {

__global__ void zero(int N, int channels, float *grid) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        for(int j=0; j<channels; j++) {
            grid[i*channels+j] = 0;
        }
    }
}

__global__ void spline_mapping_kernel_gpu(int N, int ndims, int n_neigh, int channels, const int *grid_dim_ptr, const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const int *periodic_ptr, const float *positions, const float *values, float *grid) {

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
			
			if (periodic[j]==1) {
				tmp = fmodf(tmp, 1) + (tmp < 0);
			}
			valid &= (0 <= tmp && tmp <= 1);
			shift[blockDim.x*j] = modff(tmp*(grid_dim[j] + (periodic[j]==1) - 1) + 0.5, &tmp) - 0.5;
			idx[blockDim.x*j] = tmp;
		}

		// Reduction loop over neighboring nodes
		for (int j = 0; j < n_neigh; j++) {
			int reduce = j;
			int flat = 0;
			float Wij = 1;
			for (int k = ndims - 1; k >= 0; k--) {
				int offset = -(K[k] + 1 - int(shift[blockDim.x*k] + 1)) / 2 + (reduce % (K[k] + 1));

				int in_span = grid_dim[k];
				int in_pos = idx[blockDim.x*k]+offset;

				if(periodic[k]==1) {
					flat += strides[k] * positive_modulo(in_pos, in_span);
				} else if(periodic[k]==-1) {
					int reflect = positive_modulo(in_pos, 2*(in_span-1));
					flat += strides[k] * (reflect<in_span?reflect:2*(in_span-1)-reflect); 
				} else {
					flat += strides[k] * fminf(fmaxf(in_pos, 0), in_span-1);
				}

				Wij *= kernel_gpu(shift[blockDim.x*k] - offset, K[k], dx[k], kernel_tmp)*powf(grid_dim[k]-1+(periodic[k]==1), float(dx[k]));
				reduce /= K[k] + 1;

			}
			// Accumulate contribution in each channel
			for (int k = 0; k < channels; k++) {
                atomicAdd(&grid[flat*channels + k], Wij * values[channels*i + k]);
            }
		}
	}
}

} // End extern "C"

)";


static CUmodule module;
static CUfunction kernel;
static CUfunction zero;

static bool compiled_map = false;

Status compile_map() {
	if (compiled_map) {
		return absl::OkStatus();
	}

	TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuInit(0)));
	nvrtcProgram prog;
	
	
	TF_RETURN_IF_ERROR(CudaCheckRTCCall(nvrtcCreateProgram(&prog,
		kernels_map,
		"kernels.cu",
		0,
		NULL,
		NULL)));

	TF_RETURN_IF_ERROR(CudaCheckRTCCall(nvrtcCompileProgram(prog,
		0,
		NULL)));

	size_t ptxSize;
	TF_RETURN_IF_ERROR(CudaCheckRTCCall(nvrtcGetPTXSize(prog, &ptxSize)));
	char *ptx = new char[ptxSize];
	TF_RETURN_IF_ERROR(CudaCheckRTCCall(nvrtcGetPTX(prog, ptx)));

	TF_RETURN_IF_ERROR(CudaCheckRTCCall(nvrtcDestroyProgram(&prog)));

	TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuModuleLoadDataEx(&module, ptx, 0, 0, 0)));
	TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuModuleGetFunction(&kernel, module, "spline_mapping_kernel_gpu")));
    TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuModuleGetFunction(&zero, module, "zero")));

	compiled_map = true;

	return absl::OkStatus();
}


template<typename T>
struct SplineMappingFunctor<GPU, T> {
	Status operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *values, float *output_grid) {
		TF_RETURN_IF_ERROR(compile_map());
        
        int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
        int max_order = grid.maxorder();
        int num_points = grid.num_points();
		float fill_value = grid.fill_value;
		std::vector<int> strides = grid.strides();
		std::vector<int> grid_dim = grid.dims;
		std::vector<int> K = grid.K;
		std::vector<int> dx = grid.dx;
		std::vector<int> periodic = grid.periodic;


        CUdeviceptr grid_dim_ptr, strides_ptr, K_ptr, dx_ptr, periodic_ptr;

        TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemAlloc(&grid_dim_ptr, ndims * sizeof(int))));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemAlloc(&strides_ptr, ndims * sizeof(int))));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemAlloc(&K_ptr, ndims * sizeof(int))));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemAlloc(&dx_ptr, ndims * sizeof(int))));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemAlloc(&periodic_ptr, ndims * sizeof(int))));

		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemcpyHtoD(grid_dim_ptr, grid_dim.data(), ndims * sizeof(int))));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemcpyHtoD(strides_ptr, strides.data(), ndims * sizeof(int))));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemcpyHtoD(K_ptr, K.data(), ndims * sizeof(int))));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemcpyHtoD(dx_ptr, dx.data(), ndims * sizeof(int))));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemcpyHtoD(periodic_ptr, periodic.data(), ndims * sizeof(int))));



        void *zero_args[] = {
            &num_points,
            &channels,
            &output_grid
        };

        TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuLaunchKernel(zero,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			0, NULL,
			zero_args,
			0)));

        
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
			&values,
			&output_grid
		};

		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuLaunchKernel(kernel,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			shared_size, NULL,
			args,
			0)));

        

		// Free resources
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemFree(grid_dim_ptr)));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemFree(strides_ptr)));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemFree(K_ptr)));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemFree(dx_ptr)));
		TF_RETURN_IF_ERROR(CudaCheckDriverCall(cuMemFree(periodic_ptr)));

		return absl::OkStatus();
	}
};

template struct SplineMappingFunctor<GPU, float>;