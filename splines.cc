#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include <cmath>

#include "splines.h"

REGISTER_OP("SplineGrid")
    .Input("positions: float32")
    .Input("coefficients: float32")
    .Output("interpolation: float32");




using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


template<typename Device>
class SplineGridOp : public OpKernel {
 public:
  explicit SplineGridOp(OpKernelConstruction* context) : OpKernel(context) {}


  float kernel(float x) {
    return exp(-x*x);
  }
  
  void Compute(OpKernelContext* context) override {
    const Tensor& positions = context->input(0);
    const Tensor& coefficients = context->input(1);


    
    TensorShape shape = positions.shape();
    shape.set_dim(shape.dims()-1, coefficients.dim_size(coefficients.dims()-1));
    
    
    Tensor* interpolation = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape,
                                                     &interpolation));



    auto positions_flat = positions.flat<float>();
    auto coefficients_flat = coefficients.flat<float>();
    auto interpolation_flat = interpolation->flat<float>();
    
    // COMPUTE
    int NDIMS = positions.dim_size(positions.dims()-1);
    int NCHAN = coefficients.dim_size(coefficients.dims()-1);
    int N = positions_flat.size()/NDIMS;

    /*SplineGridFunctor<Device>()(...);*/
         
  }
  
};

REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_CPU), SplineGridOp<CPUDevice>);
//REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_GPU), SplineGridOp<GPUDevice>);
// Register the GPU kernels.
//#ifdef GOOGLE_CUDA
//#define REGISTER_GPU(T)					 \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template ExampleFunctor<GPUDevice, float>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);

//#endif  // GOOGLE_CUDA
