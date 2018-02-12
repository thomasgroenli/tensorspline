#include "splines.h"

//CPU specialization of actual computation.

float kernel_cpu2(float x, int n) {
  float sigmasq = (n+1)/12.;

  return 1/sqrt(2*M_PI*sigmasq)*exp(-0.5*x*x/sigmasq);
}

void spline_grid_gradient_kernel_cpu(int N, int ndims, int n_neigh, int channels, int *grid_dim, int *strides, int *K, const float *positions, const float *grad, int *indices, float *values) {
    int *idx = new int[ndims];
    float *shift = new float[ndims];
    
    float tmp;
    int reduce;
    float Wij;
    
    for(int i=0; i<N; i++) {
      for(int j=0; j<ndims; j++) {
	tmp = positions[i*ndims+j];
	if(tmp<0 || tmp>=1) {
	  idx[j] = 0;
	  shift[j] = NAN;
	} else {
	  shift[j] = modff(tmp*(grid_dim[j]-3),&tmp);
	  idx[j] = tmp;
	}
      }

      for(int j=0; j<n_neigh; j++) {
	reduce = j;
	Wij = 1;
	for(int k=ndims-1; k>=0; k--) {
	  for(int l=0; l<channels; l++) {
	    indices[l*N*n_neigh*(ndims+1)+i*n_neigh*(ndims+1)+j*(ndims+1)+k] = idx[k]+reduce%(K[k]+1);
	  }
	  Wij *= kernel_cpu2(shift[k]+1-reduce%(K[k]+1),K[k]);
	  reduce/=K[k]+1;
	}
	for(int k=0; k<channels; k++) {
	  indices[k*N*n_neigh*(ndims+1)+i*n_neigh*(ndims+1)+j*(ndims+1)+ndims] = k;
	  values[k*N*n_neigh+i*n_neigh+j] = Wij*grad[i*channels+k];
	}
      }
    }
    delete [] idx, shift;
}

template<typename T>
struct SplineGridGradientFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d, const Grid &grid, int N, const float *positions, const float* grad, int *indices, float *values) {

    int ndims = grid.ndims();
    int n_neigh = grid.neighbors();
    int channels = grid.channels;
    std::vector<int> strides = grid.strides();
    std::vector<int> grid_dim = grid.dims;
    std::vector<int> K = grid.K;

    spline_grid_gradient_kernel_cpu(N, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), positions, grad, indices, values);

  }
     
};


template struct SplineGridGradientFunctor<Eigen::ThreadPoolDevice, float>;
