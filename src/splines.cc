#include "splines.h"
REGISTER_OP("SplineGrid")
.Input("positions: float32")
.Input("coefficients: float32")
.Attr("order: list(int) = []")
.Attr("dx: list(int) = []")
.Attr("periodic: list(int) = []")
.Attr("fill_value: float = 0")
.Attr("debug: bool = false")
.Output("interpolation: float32");


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
		int NDIMS = positions.dim_size(positions.dims() - 1);
		int NCHAN = coefficients.dim_size(coefficients.dims() - 1);
		int N = positions_flat.size() / NDIMS;
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
		for (int i = 0; i < NDIMS; i++) {
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


		int NDIMS = positions.dim_size(positions.dims() - 1);
		int NCHAN = shape.dim_size(shape.dims() - 1);
		int N = positions_flat.size() / NDIMS;
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
		for (int i = 0; i < NDIMS; i++) {
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


		int NDIMS = positions.dim_size(positions.dims() - 1);
		int NCHAN = shape.dim_size(shape.dims() - 1);
		int N = positions_flat.size() / NDIMS;
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
		for (int i = 0; i < NDIMS; i++) {
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


REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_CPU), SplineGridOp<CPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGridCoefficientGradient").Device(DEVICE_CPU), SplineGridCoefficientGradientOp<CPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGridPositionGradient").Device(DEVICE_CPU), SplineGridPositionGradientOp<CPU>);

#ifdef USE_GPU
REGISTER_KERNEL_BUILDER(Name("SplineGrid").Device(DEVICE_GPU), SplineGridOp<GPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGridCoefficientGradient").Device(DEVICE_GPU), SplineGridCoefficientGradientOp<GPU>);
REGISTER_KERNEL_BUILDER(Name("SplineGridPositionGradient").Device(DEVICE_GPU), SplineGridPositionGradientOp<GPU>);
#endif