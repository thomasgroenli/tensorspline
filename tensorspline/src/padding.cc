#include "splines.h"

REGISTER_OP("Padding")
.Input("tensor: float32")
.Attr("padding: list(int)")
.Attr("periodic: list(int)")
.Output("padded: float32");

REGISTER_OP("PaddingGradient")
.Input("gradients: float32")
.Attr("tensor_shape: shape")
.Attr("padding: list(int)")
.Attr("periodic: list(int)")
.Output("grad: float32");

template<::DeviceType Device>
class PaddingOp : public OpKernel {
private:
    std::vector<int> padding;
    std::vector<int> periodic;
public:
	explicit PaddingOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("padding", &padding);
        context->GetAttr("periodic", &periodic);
		
	}

	void Compute(OpKernelContext* context) override {
		const Tensor &tensor = context->input(0);

		TensorShape shape = tensor.shape();

        std::vector<int> out_shape;

        for(int i=0; i<shape.dims(); i++) {
		    shape.set_dim(i, shape.dim_size(i)+padding[2*i]+padding[2*i+1]);
            out_shape.push_back(shape.dim_size(i));
        }

		Tensor *padded = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, shape,
			&padded));

        auto tensor_flat = tensor.flat<float>();
		auto padded_flat = padded->flat<float>();

        PaddingFunctor<Device>()(context,
            out_shape,
			padding,
			periodic,
			tensor_flat.data(),
			padded_flat.data());

	}

};


template<::DeviceType Device>
class PaddingGradientOp : public OpKernel {
private:
	TensorShapeProto tensor_shape;
    std::vector<int> padding;
    std::vector<int> periodic;
public:
	explicit PaddingGradientOp(OpKernelConstruction* context) : OpKernel(context) {
		context->GetAttr("tensor_shape", &tensor_shape);
		context->GetAttr("padding", &padding);
        context->GetAttr("periodic", &periodic);		
	}

	void Compute(OpKernelContext* context) override {
		const Tensor &grad = context->input(0);

		TensorShape shape(tensor_shape);
		TensorShape grad_shape = grad.shape();

		Tensor *out = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, tensor_shape,
			&out));


		std::vector<int> t_shape;
		std::vector<int> g_shape;

		for(int i=0; i<shape.dims(); i++) {
            t_shape.push_back(shape.dim_size(i));
			g_shape.push_back(grad_shape.dim_size(i));
        }

		auto grad_flat = grad.flat<float>();
		auto out_flat = out->flat<float>();

		PaddingGradientFunctor<Device>()(context,
            t_shape,
			g_shape,
			padding,
			periodic,
			grad_flat.data(),
			out_flat.data());
	}
};





REGISTER_KERNEL_BUILDER(Name("Padding").Device(DEVICE_CPU), PaddingOp<CPU>);
REGISTER_KERNEL_BUILDER(Name("PaddingGradient").Device(DEVICE_CPU), PaddingGradientOp<CPU>);

#ifdef USE_GPU
REGISTER_KERNEL_BUILDER(Name("Padding").Device(DEVICE_GPU), PaddingOp<GPU>);
REGISTER_KERNEL_BUILDER(Name("PaddingGradient").Device(DEVICE_GPU), PaddingGradientOp<GPU>);
#endif