#include "splines.h"
REGISTER_OP("SplineGrid")
	.Input("positions: float32")
	.Input("coefficients: float32")
	.Attr("order: list(int) = []")
	.Attr("dx: list(int) = []")
	.Attr("fill_value: float = 0")
	.Attr("normalized: bool = true")
	.Attr("debug: bool = false")
	.Output("interpolation: float32");


REGISTER_OP("SplineGridGradient")
	.Input("positions: float32")
	.Input("gradients: float32")
	.Attr("coeff_shape: shape")
	.Attr("order: list(int) = []")
	.Attr("dx: list(int) = []")
	.Attr("normalized: bool = true")
        .Attr("debug: bool = false")
        .Output("indices: int32")
	.Output("values: float32");



template<typename Device>
class SplineGridOp : public OpKernel {
private:
    std::vector<int> K;
    std::vector<int> dx;
	float fill_value;
	bool normalized;
	bool debug;
public:
  explicit SplineGridOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("order", &K);
    context->GetAttr("dx", &dx);
	context->GetAttr("fill_value", &fill_value);
	context->GetAttr("normalized", &normalized);
	context->GetAttr("debug", &debug);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor &positions = context->input(0);
    const Tensor &coefficients = context->input(1);

    
    TensorShape shape = positions.shape();
    shape.set_dim(shape.dims()-1, coefficients.dim_size(coefficients.dims()-1));
    
    
    Tensor *interpolation = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape,
                                                     &interpolation));



    auto positions_flat = positions.flat<float>();
    auto coefficients_flat = coefficients.flat<float>();
    auto interpolation_flat = interpolation->flat<float>();
    
    // COMPUTE
    int NDIMS = positions.dim_size(positions.dims()-1);
    int NCHAN = coefficients.dim_size(coefficients.dims()-1);
    int N = positions_flat.size()/NDIMS;
    if(K.size()==0) {
      K.resize(NDIMS,DEFAULT_ORDER);
    }
    if(dx.size()==0) {
      dx.resize(NDIMS,0);
    }

    Grid grid;
    for(int i=0; i<NDIMS; i++) {
      grid.K.push_back(K[i]);
      grid.dims.push_back(coefficients.dim_size(i));
      grid.dx.push_back(dx[i]);
    }
    grid.channels = NCHAN;
	grid.fill_value = fill_value;
	grid.normalized = normalized;
    int n_neigh = grid.neighbors();
    
	OP_REQUIRES(context, positions.dim_size(positions.dims()-1) == coefficients.dims()-1,
		errors::InvalidArgument("Number of components of position tensor must agree with coefficients rank"));
    auto start = std::chrono::high_resolution_clock::now();
    SplineGridFunctor<Device>()(context,
				grid,N,
				positions_flat.data(),
				coefficients_flat.data(),
				interpolation_flat.data());
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish-start;
	if (debug) {
	  std::cout << "Interpolation performance: " << N*n_neigh*NCHAN/elapsed.count() << "C/s" << std::endl;
	}
  }

};



template<typename Device>
class SplineGridGradientOp : public OpKernel {
private:
  TensorShapeProto coeff_shape;
    std::vector<int> K;
    std::vector<int> dx;       
	bool normalized;
        bool debug;
public:
  explicit SplineGridGradientOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("coeff_shape", &coeff_shape);
    context->GetAttr("order", &K);
    context->GetAttr("dx", &dx);
	context->GetAttr("normalized", &normalized);
	context->GetAttr("debug", &debug);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& positions = context->input(0);
    const Tensor& grad = context->input(1);

    TensorShape shape(coeff_shape);
    
    auto positions_flat = positions.flat<float>();
    auto grad_flat = grad.flat<float>();

   
    int NDIMS = positions.dim_size(positions.dims()-1);
    int NCHAN = shape.dim_size(shape.dims()-1);
    int N = positions_flat.size()/NDIMS;
    if(K.size()==0) {
      K.resize(NDIMS,DEFAULT_ORDER);
    }
    if(dx.size()==0) {
      dx.resize(NDIMS,0);
    }
    
    Grid grid;
    for(int i=0; i<NDIMS; i++) {
      grid.K.push_back(K[i]);
      grid.dims.push_back(shape.dim_size(i));
      grid.dx.push_back(dx[i]);
    }
    grid.channels = NCHAN;
	grid.normalized = normalized;
    int n_neigh = grid.neighbors();

    Tensor *indices = NULL;
    Tensor *values = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, {N*n_neigh*NCHAN,NDIMS+1},
                                                     &indices));
    OP_REQUIRES_OK(context, context->allocate_output(1, {N*n_neigh*NCHAN},
						     &values));


    auto indices_flat  = indices->flat<int>();
    auto values_flat  = values->flat<float>();
    auto start = std::chrono::high_resolution_clock::now();
    SplineGridGradientFunctor<Device>()(context,
				grid,N,
				positions_flat.data(),
				grad_flat.data(),
				indices_flat.data(),
				values_flat.data());

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish-start;
    	if (debug) {
		std::cout << "Gradient performance: " << N*n_neigh*NCHAN/elapsed.count() << "C/s" << std::endl;
	}    
  }

};


REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_CPU), SplineGridOp<Eigen::ThreadPoolDevice>);
REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_GPU), SplineGridOp<Eigen::GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("SplineGridGradient").Device(DEVICE_CPU), SplineGridGradientOp<Eigen::ThreadPoolDevice>);
REGISTER_KERNEL_BUILDER(Name("SplineGridGradient").Device(DEVICE_GPU), SplineGridGradientOp<Eigen::GpuDevice>);
