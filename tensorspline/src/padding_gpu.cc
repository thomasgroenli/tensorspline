#include "splines.h"
#include <iostream>


const char *kernels_pad = R"(

extern "C" {

__global__ void padding_zero(int N, float *tensor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        tensor[i] = 0;
    }
}


__device__ int positive_modulo(int i, int n) {
	if(n==0) {
		return 0;
	}
    return (i % n + n) % n;
}

__global__ void padding_kernel_gpu(int N, int ndims, const int *out_shape_ptr, const int *strides_ptr, const int *padding_ptr, const int *periodic_ptr, const float *tensor, float *padded) {

	extern __shared__ int shared_info[];
	int *out_shape = shared_info;
	int *strides = out_shape + ndims;
	int *padding = strides + ndims;
	int *periodic = padding + 2*ndims;

	// Let leader set shared memory
	if (threadIdx.x == 0) {
		for (int i = 0; i < ndims; i++) {
			out_shape[i] = out_shape_ptr[i];
			strides[i] = strides_ptr[i];
			padding[2*i] = padding_ptr[2*i];
            padding[2*i+1] = padding_ptr[2*i+1];
			periodic[i] = periodic_ptr[i];
		}
	}

	// All threads wait for leader
	__syncthreads();

	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        int reduce = i;
        int flat = 0;
        for (int j = ndims - 1; j >= 0; j--) {
            int idx = reduce % out_shape[j];
            int in_span = out_shape[j]-padding[2*j]-padding[2*j+1];
            int in_pos = positive_modulo(idx-padding[2*j],2*(in_span-!periodic[j]));

            if(periodic[j]) {
                flat += strides[j]*((in_pos+in_span)%in_span);
            } else {
                flat += strides[j] * (in_pos<in_span?in_pos:2*(in_span-1)-in_pos); 
            }
            
            reduce /= out_shape[j];
        }
        padded[i] = tensor[flat];
    }
}


__global__ void padding_gradient_kernel_gpu(int N, int ndims, const int *grad_shape_ptr, const int *strides_ptr, const int *padding_ptr, const int *periodic_ptr, const float *grad, float *out) {

	extern __shared__ int shared_info[];
	int *grad_shape = shared_info;
	int *strides = grad_shape + ndims;
	int *padding = strides + ndims;
	int *periodic = padding + 2*ndims;

	// Let leader set shared memory
	if (threadIdx.x == 0) {
		for (int i = 0; i < ndims; i++) {
			grad_shape[i] = grad_shape_ptr[i];
			strides[i] = strides_ptr[i];
			padding[2*i] = padding_ptr[2*i];
            padding[2*i+1] = padding_ptr[2*i+1];
			periodic[i] = periodic_ptr[i];
		}
	}

	// All threads wait for leader
	__syncthreads();

	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        int reduce = i;
        int flat = 0;
        for (int j = ndims - 1; j >= 0; j--) {
            int idx = reduce % grad_shape[j];
            int in_span = grad_shape[j]-padding[2*j]-padding[2*j+1];
            int in_pos = positive_modulo(idx-padding[2*j],2*(in_span-!periodic[j]));

            if(periodic[j]) {
                flat += strides[j]*((in_pos+in_span)%in_span);
            } else {
                flat += strides[j] * (in_pos<in_span?in_pos:2*(in_span-1)-in_pos);
            }
            
            reduce /= grad_shape[j];
        }
        atomicAdd(&out[flat], grad[i]);
    }
}

} // End extern "C"

)";


static CUmodule padding_module;
static CUfunction padding_kernel;
static CUfunction padding_gradient_kernel;
static CUfunction padding_zero;

static bool compiled_pad = false;




void compile_pad(OpKernelContext *context) {
	if (compiled_pad) {
		return;
	}
	CudaCheckDriverCall(context, cuInit(0));
	nvrtcProgram prog;
	nvrtcCreateProgram(&prog,
		kernels_pad,
		"kernels_pad.cu",
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

	CudaCheckDriverCall(context, cuModuleLoadDataEx(&padding_module, ptx, 0, 0, 0));
	CudaCheckDriverCall(context, cuModuleGetFunction(&padding_kernel, padding_module, "padding_kernel_gpu"));
    CudaCheckDriverCall(context, cuModuleGetFunction(&padding_gradient_kernel, padding_module, "padding_gradient_kernel_gpu"));
    CudaCheckDriverCall(context, cuModuleGetFunction(&padding_zero, padding_module, "padding_zero"));

	compiled_pad = true;
}


template<typename T>
struct PaddingFunctor<GPU, T> {
	void operator()(OpKernelContext *context,  std::vector<int> out_shape, std::vector<int> padding, std::vector<int> periodic, const float *tensor, float *padded) {
		compile_pad(context);
        int ndims = out_shape.size();
        
        int N = 1;
		for(int i=0; i<ndims; i++) {
			N *= out_shape[i];
		}

        std::vector<int> strides(ndims, 1);
		for (int i = ndims - 2; i >= 0; i--) {
			strides[i] = strides[i + 1] * (out_shape[i + 1]-padding[2*(i+1)]-padding[2*(i+1)+1]);
		}


        int *out_shape_ptr, *strides_ptr, *padding_ptr, *periodic_ptr;

        CudaSafeCall(context, cudaMalloc(&out_shape_ptr, ndims * sizeof(int)));
		CudaSafeCall(context, cudaMalloc(&strides_ptr, ndims * sizeof(int)));
		CudaSafeCall(context, cudaMalloc(&padding_ptr, 2*ndims * sizeof(int)));
		CudaSafeCall(context, cudaMalloc(&periodic_ptr, ndims * sizeof(int)));

        CudaSafeCall(context, cudaMemcpy(out_shape_ptr, out_shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice));
		CudaSafeCall(context, cudaMemcpy(strides_ptr, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice));
		CudaSafeCall(context, cudaMemcpy(padding_ptr, padding.data(), 2*ndims * sizeof(int), cudaMemcpyHostToDevice));
		CudaSafeCall(context, cudaMemcpy(periodic_ptr, periodic.data(), ndims * sizeof(int), cudaMemcpyHostToDevice));


        int shared_size = 5 * ndims * sizeof(int);

        void *args[] = { 
			&N, 
			&ndims, 
			&out_shape_ptr, 
			&strides_ptr, 
			&padding_ptr, 
			&periodic_ptr, 
			&tensor,
            &padded};

		CudaCheckDriverCall(context, cuLaunchKernel(padding_kernel,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			shared_size, NULL,
			args,
			0));

        // Free resources
		CudaSafeCall(context, cudaFree(out_shape_ptr));
		CudaSafeCall(context, cudaFree(strides_ptr));
		CudaSafeCall(context, cudaFree(padding_ptr));
		CudaSafeCall(context, cudaFree(periodic_ptr));
    }

};

template struct PaddingFunctor<GPU, float>;

template<typename T>
struct PaddingGradientFunctor<GPU, T> {
	void operator()(OpKernelContext *context,  std::vector<int> t_shape, std::vector<int> g_shape, std::vector<int> padding, std::vector<int> periodic, const float *grad, float *out) {
		compile_pad(context);
        int ndims = t_shape.size();
        
        int N = 1;
		for(int i=0; i<ndims; i++) {
			N *= g_shape[i];
		}

        int tensor_points = 1;
		for(int i=0; i<ndims; i++) {
			tensor_points *= t_shape[i];
		}

        std::vector<int> strides(ndims, 1);
		for (int i = ndims - 2; i >= 0; i--) {
			strides[i] = strides[i + 1] * t_shape[i+1];
		}


        int *grad_shape_ptr, *strides_ptr, *padding_ptr, *periodic_ptr;

        CudaSafeCall(context, cudaMalloc(&grad_shape_ptr, ndims * sizeof(int)));
		CudaSafeCall(context, cudaMalloc(&strides_ptr, ndims * sizeof(int)));
		CudaSafeCall(context, cudaMalloc(&padding_ptr, 2*ndims * sizeof(int)));
		CudaSafeCall(context, cudaMalloc(&periodic_ptr, ndims * sizeof(int)));

        CudaSafeCall(context, cudaMemcpy(grad_shape_ptr, g_shape.data(), ndims * sizeof(int), cudaMemcpyHostToDevice));
		CudaSafeCall(context, cudaMemcpy(strides_ptr, strides.data(), ndims * sizeof(int), cudaMemcpyHostToDevice));
		CudaSafeCall(context, cudaMemcpy(padding_ptr, padding.data(), 2*ndims * sizeof(int), cudaMemcpyHostToDevice));
		CudaSafeCall(context, cudaMemcpy(periodic_ptr, periodic.data(), ndims * sizeof(int), cudaMemcpyHostToDevice));



        void *zero_args[] = {
            &tensor_points,
            &out
        };

        CudaCheckDriverCall(context, cuLaunchKernel(padding_zero,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			0, NULL,
			zero_args,
			0));

        int shared_size = 5 * ndims * sizeof(int);

        void *args[] = { 
			&N, 
			&ndims, 
			&grad_shape_ptr, 
			&strides_ptr, 
			&padding_ptr, 
			&periodic_ptr, 
			&grad,
            &out};

		CudaCheckDriverCall(context, cuLaunchKernel(padding_gradient_kernel,
			BLOCKS, 1, 1,
			THREADS, 1, 1,
			shared_size, NULL,
			args,
			0));

        // Free resources
		CudaSafeCall(context, cudaFree(grad_shape_ptr));
		CudaSafeCall(context, cudaFree(strides_ptr));
		CudaSafeCall(context, cudaFree(padding_ptr));
		CudaSafeCall(context, cudaFree(periodic_ptr));
    }

};

template struct PaddingGradientFunctor<GPU, float>;