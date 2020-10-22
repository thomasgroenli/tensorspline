#include "splines.h"

REGISTER_OP("BSpline")
.Input("x: float32")
.Attr("order: int = 1")
.Attr("dx: int = 0")
.Output("bsx: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0,c->input(0));
  return Status::OK();
});


REGISTER_OP("SplineGrid")
.Input("positions: float32")
.Input("coefficients: float32")
.Attr("order: list(int) = []")
.Attr("dx: list(int) = []")
.Attr("periodic: list(int) = []")
.Attr("fill_value: float = 0")
.Attr("debug: bool = false")
.Output("interpolation: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
::tensorflow::shape_inference::ShapeHandle positions = c->input(0);
::tensorflow::shape_inference::ShapeHandle coefficients = c->input(1);

 std::vector<::tensorflow::shape_inference::DimensionHandle> out_shape;

  int rank = c->Rank(positions);
  for(int i = 0; i<rank-1; i++) {
	out_shape.push_back(c->Dim(positions, i));
  }
  out_shape.push_back(c->Dim(coefficients,c->Rank(coefficients)-1));

  c->set_output(0, c->MakeShape(out_shape));
  return Status::OK();
});


REGISTER_OP("SplineMapping")
.Input("positions: float32")
.Input("values: float32")
.Input("weights: float32")
.Attr("grid_shape: shape")
.Attr("order: list(int) = []")
.Attr("dx: list(int) = []")
.Attr("periodic: list(int) = []")
.Attr("fill_value: float = 0")
.Attr("debug: bool = false")
.Output("grid: float32");


REGISTER_OP("SplineGridCoefficientGradient")
.Input("positions: float32")
.Input("gradients: float32")
.Attr("coeff_shape: shape")
.Attr("order: list(int) = []")
.Attr("dx: list(int) = []")
.Attr("periodic: list(int) = []")
.Attr("debug: bool = false")
.Output("indices: int32")
.Output("values: float32");

REGISTER_OP("SplineGridPositionGradient")
.Input("positions: float32")
.Input("coefficients: float32")
.Input("gradients: float32")
.Attr("order: list(int) = []")
.Attr("dx: list(int) = []")
.Attr("periodic: list(int) = []")
.Attr("debug: bool = false")
.Output("grad: float32");


template<::DeviceType Device>
class BSplineOp : public OpKernel {
private:
	int K;
	int dx;
public:
	explicit BSplineOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("order", &K);
		context->GetAttr("dx", &dx);
	}

	void Compute(OpKernelContext* context) override {
		const Tensor &x = context->input(0);

		TensorShape shape = x.shape();


		Tensor *bsx = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, shape,
			&bsx));


		auto x_flat = x.flat<float>();
		auto bsx_flat = bsx->flat<float>();

		int N = x_flat.size();

		auto start = std::chrono::high_resolution_clock::now();
		BSplineFunctor<Device>()(context,
			N,
			K,
			dx,
			x_flat.data(),
			bsx_flat.data());
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
	}

};




template<::DeviceType Device>
class SplineGridOp : public OpKernel {
private:
	std::vector<int> K;
	std::vector<int> dx;
	std::vector<int> periodic;
	float fill_value;
	bool debug;
public:
	explicit SplineGridOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("order", &K);
		context->GetAttr("dx", &dx);
		context->GetAttr("periodic", &periodic);
		context->GetAttr("fill_value", &fill_value);
		context->GetAttr("debug", &debug);
	}

	void Compute(OpKernelContext* context) override {
		const Tensor &positions = context->input(0);
		const Tensor &coefficients = context->input(1);

		TensorShape shape = positions.shape();
		shape.set_dim(shape.dims() - 1, coefficients.dim_size(coefficients.dims() - 1));


		Tensor *interpolation = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, shape,
			&interpolation));


		auto positions_flat = positions.flat<float>();
		auto coefficients_flat = coefficients.flat<float>();
		auto interpolation_flat = interpolation->flat<float>();

		// COMPUTE
		unsigned int NDIMS = positions.dim_size(positions.dims() - 1);
		unsigned int NCHAN = coefficients.dim_size(coefficients.dims() - 1);
		unsigned int N = positions_flat.size() / NDIMS;
		while (K.size() < NDIMS) {
			K.push_back(DEFAULT_ORDER);
		}
		while (dx.size() < NDIMS) {
			dx.push_back(0);
		}
		while (periodic.size() < NDIMS) {
			periodic.push_back(false);
		}

		Grid grid;
		for (unsigned int i = 0; i < NDIMS; i++) {
			grid.K.push_back(K[i]);
			grid.dims.push_back(coefficients.dim_size(i));
			grid.dx.push_back(dx[i]);
			grid.periodic.push_back(periodic[i]);
		}
		grid.channels = NCHAN;
		grid.fill_value = fill_value;
		int n_neigh = grid.neighbors();

		OP_REQUIRES(context, positions.dim_size(positions.dims() - 1) == coefficients.dims() - 1,
			errors::InvalidArgument("Number of components of position tensor must agree with coefficients rank"));
		auto start = std::chrono::high_resolution_clock::now();
		SplineGridFunctor<Device>()(context,
			grid, N,
			positions_flat.data(),
			coefficients_flat.data(),
			interpolation_flat.data());
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		if (debug) {
			std::cout << "Interpolation performance: " << N * n_neigh*NCHAN / elapsed.count() << "C/s" << std::endl;
		}
	}

};


template<::DeviceType Device>
class SplineMappingOp : public OpKernel {
private:
	TensorShapeProto grid_shape;
	std::vector<int> K;
	std::vector<int> dx;
	std::vector<int> periodic;
	float fill_value;
	bool debug;
public:
	explicit SplineMappingOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("grid_shape", &grid_shape);
		context->GetAttr("order", &K);
		context->GetAttr("dx", &dx);
		context->GetAttr("periodic", &periodic);
		context->GetAttr("fill_value", &fill_value);
		context->GetAttr("debug", &debug);
	}

	void Compute(OpKernelContext* context) override {
		const Tensor &positions = context->input(0);
		const Tensor &values = context->input(1);
		const Tensor &weights = context->input(2);

		TensorShape shape(grid_shape);
		shape.AddDim(values.dim_size(values.dims()-1));


		Tensor *output_grid = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, shape,
			&output_grid));


		auto positions_flat = positions.flat<float>();
		auto values_flat = values.flat<float>();
		auto weights_flat = weights.flat<float>();
		auto grid_flat = output_grid->flat<float>();

		// COMPUTE
		unsigned int NDIMS = positions.dim_size(positions.dims() - 1);
		unsigned int NCHAN = values.dim_size(values.dims() - 1);
		unsigned int N = positions_flat.size() / NDIMS;
		while (K.size() < NDIMS) {
			K.push_back(DEFAULT_ORDER);
		}
		while (dx.size() < NDIMS) {
			dx.push_back(0);
		}
		while (periodic.size() < NDIMS) {
			periodic.push_back(0);
		}

		Grid grid;
		for (unsigned int i = 0; i < NDIMS; i++) {
			grid.K.push_back(K[i]);
			grid.dims.push_back(output_grid->dim_size(i));
			grid.dx.push_back(dx[i]);
			grid.periodic.push_back(periodic[i]);
		}
		grid.channels = NCHAN;
		grid.fill_value = fill_value;
		int n_neigh = grid.neighbors();

		auto start = std::chrono::high_resolution_clock::now();
		SplineMappingFunctor<Device>()(context,
			grid, N,
			positions_flat.data(),
			values_flat.data(),
			weights_flat.data(),
			grid_flat.data());
		
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		if (debug) {
			std::cout << "Interpolation performance: " << N * n_neigh*NCHAN / elapsed.count() << "C/s" << std::endl;
		}
	}

};

template<::DeviceType Device>
class SplineGridCoefficientGradientOp : public OpKernel {
private:
	TensorShapeProto coeff_shape;
	std::vector<int> K;
	std::vector<int> dx;
	std::vector<int> periodic;
	bool debug;
public:
	explicit SplineGridCoefficientGradientOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("coeff_shape", &coeff_shape);
		context->GetAttr("order", &K);
		context->GetAttr("dx", &dx);
		context->GetAttr("periodic", &periodic);
		context->GetAttr("debug", &debug);
	}

	void Compute(OpKernelContext* context) override {
		const Tensor& positions = context->input(0);
		const Tensor& grad = context->input(1);

		TensorShape shape(coeff_shape);

		auto positions_flat = positions.flat<float>();
		auto grad_flat = grad.flat<float>();


		unsigned int NDIMS = positions.dim_size(positions.dims() - 1);
		unsigned int NCHAN = shape.dim_size(shape.dims() - 1);
		unsigned int N = positions_flat.size() / NDIMS;
		while (K.size() < NDIMS) {
			K.push_back(DEFAULT_ORDER);
		}
		while (dx.size() < NDIMS) {
			dx.push_back(0);
		}
		while (periodic.size() < NDIMS) {
			periodic.push_back(0);
		}


		Grid grid;
		for (unsigned int i = 0; i < NDIMS; i++) {
			grid.K.push_back(K[i]);
			grid.dims.push_back(shape.dim_size(i));
			grid.dx.push_back(dx[i]);
			grid.periodic.push_back(periodic[i]);
		}
		grid.channels = NCHAN;
		int n_neigh = grid.neighbors();

		Tensor *indices = NULL;
		Tensor *values = NULL;

		OP_REQUIRES_OK(context, context->allocate_output(0, { N*n_neigh*NCHAN,NDIMS + 1 },
			&indices));
		OP_REQUIRES_OK(context, context->allocate_output(1, { N*n_neigh*NCHAN },
			&values));


		auto indices_flat = indices->flat<int>();
		auto values_flat = values->flat<float>();
		auto start = std::chrono::high_resolution_clock::now();
		SplineGridCoefficientGradientFunctor<Device>()(context,
			grid, N,
			positions_flat.data(),
			grad_flat.data(),
			indices_flat.data(),
			values_flat.data());

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		if (debug) {
			std::cout << "Gradient performance: " << N * n_neigh*NCHAN / elapsed.count() << "C/s" << std::endl;
		}
	}

};

template<::DeviceType Device>
class SplineGridPositionGradientOp : public OpKernel {
private:
	std::vector<int> K;
	std::vector<int> dx;
	std::vector<int> periodic;
	bool debug;
public:
	explicit SplineGridPositionGradientOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("order", &K);
		context->GetAttr("dx", &dx);
		context->GetAttr("periodic", &periodic);
		context->GetAttr("debug", &debug);
	}

	void Compute(OpKernelContext* context) override {
		const Tensor &positions = context->input(0);
		const Tensor &coefficients = context->input(1);
		const Tensor& grad = context->input(2);


		auto positions_flat = positions.flat<float>();
		auto coefficients_flat = coefficients.flat<float>();
		auto grad_flat = grad.flat<float>();

		auto shape = coefficients.shape();


		unsigned int NDIMS = positions.dim_size(positions.dims() - 1);
		unsigned int NCHAN = shape.dim_size(shape.dims() - 1);
		unsigned int N = positions_flat.size() / NDIMS;
		while (K.size() < NDIMS) {
			K.push_back(DEFAULT_ORDER);
		}
		while (dx.size() < NDIMS) {
			dx.push_back(0);
		}
		while (periodic.size() < NDIMS) {
			periodic.push_back(0);
		}


		Grid grid;
		for (unsigned int i = 0; i < NDIMS; i++) {
			grid.K.push_back(K[i]);
			grid.dims.push_back(shape.dim_size(i));
			grid.dx.push_back(dx[i]);
			grid.periodic.push_back(periodic[i]);
		}
		grid.channels = NCHAN;
		int n_neigh = grid.neighbors();

		Tensor *result = NULL;

		OP_REQUIRES_OK(context, context->allocate_output(0, positions.shape(),
			&result));


		auto result_flat = result->flat<float>();
		auto start = std::chrono::high_resolution_clock::now();
		SplineGridPositionGradientFunctor<Device>()(context,
			grid, N,
			positions_flat.data(),
			coefficients_flat.data(),
			grad_flat.data(),
			result_flat.data());

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		if (debug) {
			std::cout << "Gradient performance: " << N * n_neigh*NCHAN / elapsed.count() << "C/s" << std::endl;
		}
	}

};


REGISTER_KERNEL_BUILDER(Name("BSpline").Device(DEVICE_CPU), BSplineOp<CPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_CPU), SplineGridOp<CPU>);
REGISTER_KERNEL_BUILDER(Name("SplineMapping").Device(DEVICE_CPU), SplineMappingOp<CPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGridCoefficientGradient").Device(DEVICE_CPU), SplineGridCoefficientGradientOp<CPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGridPositionGradient").Device(DEVICE_CPU), SplineGridPositionGradientOp<CPU>);

#ifdef USE_GPU
REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_GPU), SplineGridOp<GPU>);
REGISTER_KERNEL_BUILDER(Name("SplineMapping").Device(DEVICE_GPU), SplineMappingOp<GPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGridCoefficientGradient").Device(DEVICE_GPU), SplineGridCoefficientGradientOp<GPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGridPositionGradient").Device(DEVICE_GPU), SplineGridPositionGradientOp<GPU>);
#endif


extern "C" {
	bool cuda_enabled() {
		#ifdef USE_GPU
			return true;
		#else
	    	return false;
		#endif
    }

    void set_launch_config(int threads, int blocks) {
	    THREADS = threads;
	    BLOCKS = blocks;
    }
};
