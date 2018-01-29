#include "splines.h"
REGISTER_OP("SplineGrid")
    .Input("positions: float32")
    .Input("coefficients: float32")
    .Output("interpolation: float32");

REGISTER_OP("SplineGridGradient")
    .Input("positions: float32")
    .Input("coeff_shape: int32")
    .Output("indices: int32")
    .Output("values: float32");




using namespace tensorflow;

template<typename Device>
class SplineGridOp : public OpKernel {
 public:
  explicit SplineGridOp(OpKernelConstruction* context) : OpKernel(context) {}

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

    Grid grid;
    for(int i=0; i<NDIMS; i++) {
      grid.K.push_back(3);
      grid.dims.push_back(coefficients.dim_size(i));
    }
    grid.channels = NCHAN;
    auto start = std::chrono::high_resolution_clock::now();
    SplineGridFunctor<Device>()(context->eigen_device<Device>(),
				grid,N,
				positions_flat.data(),
				coefficients_flat.data(),
				interpolation_flat.data());
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish-start;
    std::cout << "Computation took: " << elapsed.count() << " s" << std::endl;
  }

};



template<typename Device>
class SplineGridGradientOp : public OpKernel {
 public:
  explicit SplineGridGradientOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& positions = context->input(0);
    const Tensor& coeff_shape = context->input(1);
    
    Tensor* indices = NULL;
    Tensor* values = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, coeff_shape,
                                                     &indices));


  }

};


REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_CPU), SplineGridOp<Eigen::ThreadPoolDevice>);
REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_GPU), SplineGridOp<Eigen::GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("SplineGridGradient").Device(DEVICE_CPU), SplineGridGradientOp<Eigen::ThreadPoolDevice>);
//REGISTER_KERNEL_BUILDER(Name("SplineGridGradient").Device(DEVICE_GPU), SplineGridGradientOp<Eigen::GpuDevice>);
