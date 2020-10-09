#pragma once

#include <cuda.h>
#include <stdexcept>

inline void CudaSafeCall( cudaError err )
{
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}


inline void CudaCheckDriverCall(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        const char **error_string;
        cuGetErrorString(result, error_string);
        throw std::runtime_error(*error_string);
    }
}