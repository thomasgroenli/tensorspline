#pragma once

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <cuda.h>
#include <stdexcept>

using namespace tensorflow;

inline void CudaSafeCall(OpKernelContext *context, cudaError err )
{
    OP_REQUIRES(context, err == cudaSuccess, errors::Unknown(cudaGetErrorString(err)));
}


inline void CudaCheckDriverCall(OpKernelContext *context, CUresult result)
{
    const char *error_string;
    cuGetErrorString(result, &error_string);
    OP_REQUIRES(context, result == CUDA_SUCCESS, errors::Unknown(error_string));
}