#include "splines.h"

//CPU specialization of actual computation.

float kernel_cpu(float x, int n, int dx) {
  float sigmasq = (n+1)/12.;
  
  float a = 1/sqrt(2*M_PI*sigmasq)*exp(-0.5*x*x/sigmasq),b=0;
  for(int n=1; n<=dx; n++) {
    a = -(x*a+(n-1)*b)/sigmasq;
    b = a;
  }
  return a;
}

void spline_grid_kernel_cpu(int N, int ndims, int n_neigh, int channels, const int *grid_dim, const int *strides, const int *K, const int *dx, const float *positions, const float *coefficients, float *out) {
    int *idx = new int[ndims];
    float *shift = new float[ndims];
    float *channel_sum = new float[channels];

    
    float tmp;
    int reduce;
    int flat;
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
      for(int j=0; j<channels; j++) {
	channel_sum[j] = 0;
      }

      for(int j=0; j<n_neigh; j++) {
	reduce = j;
	flat = 0;
	Wij = 1;
	for(int k=ndims-1; k>=0; k--) {
	  flat += strides[k]*(idx[k]+reduce%(K[k]+1));
	  Wij *= kernel_cpu(shift[k]+1-reduce%(K[k]+1),K[k],dx[k]);
	  reduce/=K[k]+1;
	}
	for(int k=0; k<channels; k++) {
	  channel_sum[k] += Wij*coefficients[channels*flat+k];
	}
      }
      for(int j=0; j<channels; j++) {
	out[i*channels+j] = channel_sum[j];
      }
    }
    delete [] idx, shift, channel_sum;
}

template<typename T>
struct SplineGridFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d, const Grid &grid, int N, const float *positions, const float *coefficients, float *out) {

    int ndims = grid.ndims();
    int n_neigh = grid.neighbors();
    int channels = grid.channels;
    std::vector<int> strides = grid.strides();
    std::vector<int> grid_dim = grid.dims;
    std::vector<int> K = grid.K;
    std::vector<int> dx = grid.dx;
    spline_grid_kernel_cpu(N, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), dx.data(), positions, coefficients, out);

  }
     
};



template struct SplineGridFunctor<Eigen::ThreadPoolDevice, float>;

