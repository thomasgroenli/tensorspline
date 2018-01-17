#include "splines.h"

// CPU specialization of actual computation.
struct SplineGridFunctor<CPUDevice> {
  void operator()(const CPUDevice& d, int NDIMS, int NCHAN, int N, const float *positions, const float *coefficients, float *out) {
    
    int *strides = new int[NDIMS];
    int *grid_dim = new int[NDIMS];
    
    int *idx = new int[NDIMS];
    float *shift = new float[NDIMS];
    float *channel_sum = new float[NCHAN];

    
    double tmp;
    int reduce;
    int flat;
    int radius;
    
    int n_neigh = pow(4,NDIMS);


    strides[NDIMS-1] = 1;
    grid_dim[NDIMS-1] = coefficients.dim_size(NDIMS-1);
    for(int i=NDIMS-2; i>=0; i--) {
      grid_dim[i] = coefficients.dim_size(i);
      strides[i] = strides[i+1]*coefficients.dim_size(i);
    }
    
    for(int i=0; i<N; i++) {
      for(int j=0; j<NDIMS; j++) {
	shift[j] = modf(positions_flat(i*NDIMS+j)*(grid_dim[j]-3),&tmp);
	idx[j] = tmp;
      }
      for(int j=0; j<NCHAN; j++) {
	channel_sum[j] = 0;
      }
      for(int j=0; j<n_neigh; j++) {
	reduce = j;
	flat = 0;
	radius = 0;
	for(int k=NDIMS-1; k>=0; k--) {
	  flat += strides[k]*(idx[k]-1+reduce%4);
	  radius += (shift[k]+1-reduce%4)*(shift[k]+1-reduce%4);
	  reduce/=4;
	}

	for(int k=0; k<NCHAN; k++) {
	  channel_sum[k] += coefficients_flat(NCHAN*flat+k)*kernel(radius);
	}
      }
      for(int j=0; j<NCHAN; j++) {
	interpolation_flat(i*NCHAN+j) = channel_sum[j];
      }
    }
  }
};
