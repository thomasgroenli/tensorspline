#pragma once
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdio.h>


#define DEFAULT_ORDER 3

struct Grid {
  std::vector<int> K; // Spline orders
  std::vector<int> dims;
  std::vector<int> dx;
  int channels;
  

  int ndims() const {
    return K.size();
  }
  std::vector<int> strides() const {
    std::vector<int> result(ndims(),1);
    for(int i=ndims()-2; i>=0; i--) {
      result[i] = result[i+1]*dims[i+1];
    }
    return result;
  }
  int neighbors() const {
    int result = 1;
    for(int i=0; i<ndims(); i++) {
      result *= K[i]+1;
    }
    return result;
  }
  void debug() {
    std::cout << "NDims: " << ndims() << std::endl;

    std::cout << "Dims: ";
    for(int i=0; i<ndims(); i++) {
      std::cout << dims[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Strides: ";
    auto strides_ = strides();
    for(int i=0; i<ndims(); i++) {
      std::cout << strides_[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Order: ";
    for(int i=0; i<ndims(); i++) {
      std::cout << K[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Neighbors: " << neighbors() << std::endl;
    
    std::cout << "Diff: ";
    for(int i=0; i<ndims(); i++) {
      std::cout << dx[i] << " ";
    }
    std::cout << std::endl;
  }
};

template<typename Device, typename T=float>
struct SplineGridFunctor {
  void operator()(const Device& d, const Grid &, int, const float *, const float *, float *);
};

template<typename Device, typename T=float>
struct SplineGridGradientFunctor {
  void operator()(const Device& d, const Grid &, int, const float *, const float *, int *, float *);
};

