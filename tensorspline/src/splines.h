#pragma once
#define _USE_MATH_DEFINES

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdio.h>

using namespace tensorflow;
#define DEFAULT_ORDER 3

static int THREADS = 128;
static int BLOCKS = 1024;

typedef std::atomic_bool lock;

struct Grid {
	std::vector<int> K; // Spline orders
	std::vector<int> dims;
	std::vector<int> dx;
	std::vector<int> periodic;
	float fill_value;
	int channels;


	int ndims() const {
		return K.size();
	}
	int num_points() const {
		int product = 1;
		for(int i=0; i<ndims(); i++) {
			product *= dims[i];
		}
		return product;
	}
	std::vector<int> strides() const {
		std::vector<int> result(ndims(), 1);
		for (int i = ndims() - 2; i >= 0; i--) {
			result[i] = result[i + 1] * dims[i + 1];
		}
		return result;
	}
	int neighbors() const {
		int result = 1;
		for (int i = 0; i < ndims(); i++) {
			result *= K[i] + 1;
		}
		return result;
	}
	int maxorder() const {
		int result = 0;
		for (int i = 0; i < ndims(); i++) {
			result = K[i] > result ? K[i] : result;
		}
		return result;
	}
	void debug() const {
		std::cout << "NDims: " << ndims() << std::endl;

		std::cout << "Dims: ";
		for (int i = 0; i < ndims(); i++) {
			std::cout << dims[i] << " ";
		}
		std::cout << std::endl;

		std::cout << "Strides: ";
		auto strides_ = strides();
		for (int i = 0; i < ndims(); i++) {
			std::cout << strides_[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "Channels: " << channels << std::endl;
		std::cout << "Order: ";
		for (int i = 0; i < ndims(); i++) {
			std::cout << K[i] << " ";
		}
		std::cout << std::endl;

		std::cout << "Neighbors: " << neighbors() << std::endl;

		std::cout << "Diff: ";
		for (int i = 0; i < ndims(); i++) {
			std::cout << dx[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "Max order: " << maxorder() << std::endl;
		std::cout << "Fill value: " << fill_value << std::endl;
	}
};

enum DeviceType {CPU, GPU};

float kernel_cpu(float x, int p, int dx, float *tmp);

template<::DeviceType Device, typename T = float>
struct PaddingFunctor {
	void operator()(OpKernelContext *, std::vector<int>, std::vector<int>, std::vector<int>, const float *, float *);
};
template<::DeviceType Device, typename T = float>
struct PaddingGradientFunctor {
	void operator()(OpKernelContext *, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, const float *, float *);
};

template<::DeviceType Device, typename T = float>
struct BSplineFunctor {
	void operator()(OpKernelContext *, int, int, int, const float *, float *);
};

template<::DeviceType Device, typename T = float>
struct SplineGridFunctor {
	void operator()(OpKernelContext *, const Grid &, int, const float *, const float *, float *);
};

template<::DeviceType Device, typename T = float>
struct SplineGridCoefficientGradientFunctor {
	void operator()(OpKernelContext *, const Grid &, int, const float *, const float *, int *, float *);
};

template<::DeviceType Device, typename T = float>
struct SplineGridPositionGradientFunctor {
	void operator()(OpKernelContext *, const Grid &, int, const float *, const float *, const float *, float *);
};

template<::DeviceType Device, typename T = float>
struct SplineMappingFunctor {
	void operator()(OpKernelContext *, const Grid &, int, const float *, const float *, const float *, float *);
};

inline int positive_modulo(int i, int n) {
    if(n==0) {
        return 0;
    }
    return (i % n + n) % n;
}