#include "splines.h"
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define CUDART_NAN_F            __int_as_float(0x7fffffff)
#define THREADS 64

//GPU specialization of actual computation.

__device__
float kernel_gpu2(float x, int n, int dx) {
  float sigmasq = (n+1)/12.;
  float prefactor = 1;
  if(dx == 1) {
    prefactor = -x/sigmasq;
  } else if(dx == 2) {
    prefactor = (x*x-sigmasq)/(sigmasq*sigmasq);
  }
  return prefactor/sqrt(2*M_PI*sigmasq)*exp(-0.5*x*x/sigmasq);
}


__global__ void spline_grid_gradient_kernel_gpu(int N, int ndims, int n_neigh, int channels, const int *grid_dim_ptr,const int *strides_ptr, const int *K_ptr, const int *dx_ptr, const float *positions, const float *grad, int *indices, float *values) {

  extern __shared__ int shared_info[];
  int *grid_dim = shared_info;
  int *strides = grid_dim+ndims;
  int *K = strides+ndims;
  int *dx = K+ndims;

  // Let leader set shared memory
  if(threadIdx.x==0) {
    for(int i=0; i<ndims; i++) {
      grid_dim[i] = grid_dim_ptr[i];
      strides[i] = strides_ptr[i];
      K[i] = K_ptr[i];
      dx[i] = dx_ptr[i];
      //      printf("%i: %i, %i, %i\n",i, grid_dim[i], strides[i], K[i]); // Debug
    }
  }

  // All threads wait for leader
  __syncthreads();

  // Stride into shared memory
  int *idx = dx+ndims+threadIdx.x;
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
	  indices[i*n_neigh*channels*(ndims+1)+j*channels*(ndims+1)+l*(ndims+1)+k] = idx[blockDim.x*k]+reduce%(K[k]+1);
	}
	Wij *= kernel_gpu2(shift[blockDim.x*k]+1-reduce%(K[k]+1),K[k],dx[k]);
	reduce/=K[k]+1;
      }
      for(int k=0; k<channels; k++) {
	indices[i*n_neigh*channels*(ndims+1)+j*channels*(ndims+1)+k*(ndims+1)+ndims] = k;
	values[i*n_neigh*channels+j*channels+k] = Wij*grad[i*channels+k];
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
    std::vector<int> dx = grid.dx;

    int *grid_dim_ptr, *strides_ptr, *K_ptr, *dx_ptr;

    cudaMalloc(&grid_dim_ptr, ndims*sizeof(int));
    cudaMalloc(&strides_ptr, ndims*sizeof(int));
    cudaMalloc(&K_ptr, ndims*sizeof(int));
    cudaMalloc(&dx_ptr, ndims*sizeof(int));

    cudaMemcpy(grid_dim_ptr, grid_dim.data(), ndims*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(strides_ptr, strides.data(), ndims*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(K_ptr, K.data(), ndims*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dx_ptr, dx.data(), ndims*sizeof(int), cudaMemcpyHostToDevice);

    // Compute shared memory size
    int shared_size = 4*ndims*sizeof(int);
    shared_size += ndims*THREADS*sizeof(int);
    shared_size += ndims*THREADS*sizeof(float);

    // Enqueue kernel
    spline_grid_gradient_kernel_gpu<<<80, THREADS, shared_size>>>(N, ndims, n_neigh, channels, grid_dim_ptr, strides_ptr, K_ptr, dx_ptr, positions, grad, indices, values);

    // Free resources
    cudaFree(grid_dim_ptr);
    cudaFree(strides_ptr);
    cudaFree(K_ptr);
    cudaFree(dx_ptr);
  }
     
};


template struct SplineGridGradientFunctor<Eigen::GpuDevice,float>;
