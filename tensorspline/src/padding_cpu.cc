#include "splines.h"



void padding_kernel_cpu(int start, int end, int ndims, int *out_shape, int *strides, int *padding, int *periodic, const float *tensor, float *padded) {
    for (int i = start; i < end; ++i) {
		int reduce = i;
        int flat = 0;
        for (int j = ndims - 1; j >= 0; j--) {
            int idx = reduce % out_shape[j];
            int in_pos = idx-padding[2*j];
            int in_span = out_shape[j]-padding[2*j]-padding[2*j+1];

            if(periodic[j]) {
                flat += strides[j]*((in_pos+in_span)%in_span);
            } else {
                flat += strides[j] * fmin(fmax(in_pos, 0), in_span-1);
            }
            
            reduce /= out_shape[j];
        }
        padded[i] = tensor[flat];
    } 
}


template<typename T>
struct PaddingFunctor<CPU, T> {
	void operator()(OpKernelContext *context,  std::vector<int> out_shape, std::vector<int> padding, std::vector<int> periodic, const float *tensor, float *padded) {
		int ndims = out_shape.size();
        
        int N = 1;
		for(int i=0; i<ndims; i++) {
			N *= out_shape[i];
		}

        std::vector<int> strides(ndims, 1);
		for (int i = ndims - 2; i >= 0; i--) {
			strides[i] = strides[i + 1] * (out_shape[i + 1]-padding[2*(i+1)]-padding[2*(i+1)+1]);
		}

#ifdef USE_MULTITHREAD
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(pool->NumThreads(), pool, N, 1024, [&](int start, int end) {
			padding_kernel_cpu(start, end, out_shape.size(), out_shape.data(), strides.data(), padding.data(), periodic.data(), tensor, padded);
		});
#else
        padding_kernel_cpu(0, N, out_shape.size(), out_shape.data(), strides.data(), padding.data(), periodic.data(), tensor, padded);
#endif
    }
};

template struct PaddingFunctor<CPU, float>;



void padding_gradient_kernel_cpu(int start, int end, int ndims, int *grad_shape, int *strides, int *padding, int *periodic, const float *tensor, const float *grad, float *out, lock *locks) {    
    bool lock_var = false;
    for (int i = start; i < end; ++i) {
		int reduce = i;
        int flat = 0;
        for (int j = ndims - 1; j >= 0; j--) {
            int idx = reduce % grad_shape[j];
            int in_pos = idx-padding[2*j];
            int in_span = grad_shape[j]-padding[2*j]-padding[2*j+1];

            if(periodic[j]) {
                flat += strides[j]*((in_pos+in_span)%in_span);
            } else {
                flat += strides[j] * fmin(fmax(in_pos, 0), in_span-1);
            }
            
            reduce /= grad_shape[j];
        }
        while(!locks[flat].compare_exchange_strong(lock_var, true)) {lock_var=false;}
        out[flat] += grad[i];
        locks[flat].store(false);
    } 
}

template<typename T>
struct PaddingGradientFunctor<CPU, T> {
	void operator()(OpKernelContext *context,  std::vector<int> t_shape, std::vector<int> g_shape, std::vector<int> padding, std::vector<int> periodic, const float *tensor, const float *grad, float *out) {
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


        lock *locks = new lock[tensor_points];
        for(int i=0; i<tensor_points; i++) {
            locks[i] = false;
            out[i] = 0;
        }

#ifdef USE_MULTITHREAD
		auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
		Shard(pool->NumThreads(), pool, N, 1024, [&](int start, int end) {
			padding_gradient_kernel_cpu(start, end, ndims, g_shape.data(), strides.data(), padding.data(), periodic.data(), tensor, grad, out, locks);
        });
#else
        padding_gradient_kernel_cpu(0, N, ndims, g_shape.data(), strides.data(), padding.data(), periodic.data(), tensor, grad, out, locks);
        
#endif

        delete[] locks;
    }
};

template struct PaddingGradientFunctor<CPU, float>;

