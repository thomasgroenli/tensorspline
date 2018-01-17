#include "splines.h"

// GPU specialization of actual computation.
struct SplineGridFunctor<GPUDevice> {
  void operator()(const CPUDevice& d, int NDIMS, int NCHAN, int N, const float *positions, const float *coefficients, const float *out) {
    
  }
};
