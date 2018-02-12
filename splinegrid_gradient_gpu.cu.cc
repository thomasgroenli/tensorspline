#include "splines.h"
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define CUDART_NAN_F            __int_as_float(0x7fffffff)
#define THREADS 128

//GPU specialization of actual computation.

__device__
float kernel_gpu2(float x, int n) {
  float sigmasq = (n+1)/12.;
  return 1/sqrtf(2*M_PI*sigmasq)*expf(-0.5*x*x/sigmasq);
}


__global__ void spline_grid_gradient_kernel_gpu(int N, int ndims, int n_neigh, int channels, int *grid_dim_ptr, int *strides_ptr, int *K_ptr, const float *positions, const float *grad, int *indices, float *values) {

  extern __shared__ int shared_info[];
  int *grid_dim = shared_info;
  int *strides = grid_dim+ndims;
  int *K = strides+ndims;

  // Let leader set shared memory
  if(threadIdx.x==0) {
    for(int i=0; i<ndims; i++) {
      grid_dim[i] = grid_dim_ptr[i];
      strides[i] = strides_ptr[i];
      K[i] = K_ptr[i];
      //      printf("%i: %i, %i, %i\n",i, grid_dim[i], strides[i], K[i]); // Debug
    }
  }

  // All threads wait for leader
  __syncthreads();

  // Stride into shared memory
  int *idx = K+ndims+threadIdx.x;
  float *shift = (float*)(idx+ndims*blockDim.x);

  // Accumulating variables
  float tmp;
  int reduce;
  float Wij;

  // grid-stride loop
  for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<N; i += blockDim.x * gridDim.x) {

    // Fetch the main index and shift
    for(int j=0; j<ndims; j++) {
      tmp = positions[i*ndims+j];
      if(tmp<0 || tmp>=1) { // Outside bounds
	idx[blockDim.x*j] = 0;
	shift[blockDim.x*j] = CUDART_NAN_F;
      } else {
	shift[blockDim.x*j] = modff(tmp*(grid_dim[j]-3),&tmp);
	idx[blockDim.x*j] = tmp;
      }
    }

    // Reduction loop over neighboring nodes
    for(int j=0; j<n_neigh; j++) {
      reduce = j;
      Wij = 1;
      for(int k=ndims-1; k>=0; k--) {
	for(int l=0; l<channels; l++) {
	  indices[l*N*n_neigh*(ndims+1)+i*n_neigh*(ndims+1)+j*(ndims+1)+k] = idx[blockDim.x*k]+reduce%(K[k]+1);
	}
	Wij *= kernel_gpu2(shift[blockDim.x*k]+1-reduce%(K[k]+1),K[k]);
	reduce/=K[k]+1;
      }
      for(int k=0; k<channels; k++) {
	indices[k*N*n_neigh*(ndims+1)+i*n_neigh*(ndims+1)+j*(ndims+1)+ndims] = k;
	values[k*N*n_neigh+i*n_neigh+j] = Wij*grad[i*channels+k];
      }
    }
  }
}

template<typename T>
struct SplineGridGradientFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const Grid &grid, int N, const float *positions, const float *grad, int *indices, float *values) {

    
    int ndims = grid.ndims();
    int n_neigh = grid.neighbors();
    int channels = grid.channels;
    std::vector<int> strides = grid.strides();
    std::vector<int> grid_dim = grid.dims;
    std::vector<int> K = grid.K;

    int *grid_dim_ptr, *strides_ptr, *K_ptr;
    
    cudaMallocManaged(&grid_dim_ptr, ndims*sizeof(int));
    cudaMallocManaged(&strides_ptr, ndims*sizeof(int));
    cudaMallocManaged(&K_ptr, ndims*sizeof(int));
    for(int i=0; i<ndims; i++) {
      grid_dim_ptr[i] = grid_dim[i];
      strides_ptr[i] = strides[i];
      K_ptr[i] = K[i];
    }


    // Compute shared memory size
    int shared_size = 3*ndims*sizeof(int);
    shared_size += ndims*THREADS*sizeof(int);
    shared_size += ndims*THREADS*sizeof(float);

    // Enqueue kernel
    spline_grid_gradient_kernel_gpu<<<80, THREADS, shared_size>>>(N, ndims, n_neigh, channels, grid_dim_ptr, strides_ptr, K_ptr, positions, grad, indices, values);

    // Free resources
    cudaFree(grid_dim_ptr);
    cudaFree(strides_ptr);
    cudaFree(K_ptr);
  }
     
};


template struct SplineGridGradientFunctor<Eigen::GpuDevice,float>;
