#define EIGEN_USE_GPU

#include "splines.h"
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
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

extern "C" {

__global__ void zero(int N, int channels, float *grid) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        for(int j=0; j<channels; j++) {
            grid[i*channels+j] = 0;
        }
    }
}

__global__ void normalize(int N, int channels, float *grid, float *density) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        for(int j=0; j<channels; j++) {
            grid[i*channels+j] /= density[i*channels+j];
        }
    }
}

__global__ void spline_mapping_kernel_gpu(int N, int ndims, int n_neigh, int channels, const int *grid_dim_ptr, const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const int *periodic_ptr, const float *positions, const float *values, const float *weights, float *grid, float *density) {

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
                atomicAdd(&grid[flat*channels + k], weights[channels*i + k] * Wij * values[channels*i + k]);
				atomicAdd(&density[flat*channels + k], weights[channels*i + k] * Wij);
            }
		}
	}
}

} // End extern "C"

)";


static CUmodule module;
static CUfunction kernel;
static CUfunction zero;
static CUfunction normalize;

static bool compiled_map = false;

void compile_map() {
	if (compiled_map) {
		return;
	}
	cuInit(0);
	nvrtcProgram prog;
	nvrtcCreateProgram(&prog,
		kernels_map,
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
	cuModuleGetFunction(&kernel, module, "spline_mapping_kernel_gpu");
    cuModuleGetFunction(&zero, module, "zero");
    cuModuleGetFunction(&normalize, module, "normalize");

	compiled_map = true;
}



template<typename T>
struct SplineMappingFunctor<GPU, T> {
	void operator()(OpKernelContext *context, const Grid &grid, int N, const float *positions, const float *values, const float *weights, float *output_grid) {
		compile_map();
        
        int ndims = grid.ndims();
		int n_neigh = grid.neighbors();
		int channels = grid.channels;
        int max_order = grid.maxorder();
        int num_points = grid.num_points();
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


        float *density;
        cudaMalloc(&density, num_points * channels * sizeof(float));


        int zero_chan;
        void *zero_args[] = {
            &num_points,
            &zero_chan,
            nullptr
        };

        zero_chan = channels;
        zero_args[2] = &output_grid;
        cuLaunchKernel(zero,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			0, NULL,
			zero_args,
			0);

        zero_chan = 1;
        zero_args[2] = &density;
        cuLaunchKernel(zero,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			0, NULL,
			zero_args,
			0);

        
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
            &weights, 
			&output_grid,
            &density};

		cuLaunchKernel(kernel,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			shared_size, NULL,
			args,
			0);

        void *normalize_args[] = {
            &num_points, 
            &channels, 
            &output_grid, 
            &density
        };

        cuLaunchKernel(normalize,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			0, NULL,
			normalize_args,
			0);

		// Free resources
		cudaFree(grid_dim_ptr);
		cudaFree(strides_ptr);
		cudaFree(K_ptr);
		cudaFree(dx_ptr);
		cudaFree(periodic_ptr);
        cudaFree(density);
	}
};

template struct SplineMappingFunctor<GPU, float>;