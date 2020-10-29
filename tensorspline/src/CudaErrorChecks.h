#pragma once

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <cuda.h>
#include <stdexcept>

using namespace tensorflow;

inline Status CudaSafeCall(cudaError err)
{
    if(err != cudaSuccess) {
        return errors::Unknown(cudaGetErrorString(err));
    }
    return Status::OK();
}


inline Status CudaCheckDriverCall(CUresult result)
{
    const char *error_string;
    cuGetErrorString(result, &error_string);
    if(result != CUDA_SUCCESS) {
        return errors::Unknown(error_string);
    }
    return Status::OK();
}

inline Status CudaCheckRTCCall(nvrtcResult result)
{
    if(result != NVRTC_SUCCESS) {
        return errors::Unknown(nvrtcGetErrorString(result));
    }
    return Status::OK();
}