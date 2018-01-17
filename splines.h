#ifndef SPLINES_H_
#define SPLINES_H_

template<typename Device>
struct SplineGridFunctor {
  void operator()(const Device& d, int, int, int, const float *, const float *, float*);
};

/*#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template<typename Eigen::GpuDevice>
struct SplineGridFunctor {
  void operator()(const Eigen::GpuDevice& d, int, int, int, const float *, const float *, float*);
};
#endif
*/
#endif SPLINES_H_
