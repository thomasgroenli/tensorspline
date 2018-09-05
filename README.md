# TensorSpline

## Introduction

This package provides support for N-D B-spline interpolation and gradients of arbitrary orders and differentials in TensorFlow.

To contribute to this project, talk to Thomas Grønli (thomas.gronli@gmail.com)

## Usage
### Installation
Install the precompiled wheel found under the "Downloads"-section
```console
$ pip install <TensorSpline>.whl
```

OR build from source.

### Reference
The available function has the following signature  

- ``interpolation = spline_grid(positions, coefficients, order=[], dx=[], fill_value=0, normalized=True)``

- ``positions``: Query positions, Tensor or Tensor-like object with shape ``[SHAPE, d]``

- ``coefficients``: Coefficient grid, Tensor or Tensor-like object with shape ``[N1, N2, ..., Nd, channels]``

- ``order``: Interpolation orders, list of ints length <= d, defaults to 3

- ``dx``: Differentiation orders, list of ints length <= d, defaults to 0

- ``fill_value``: Fill value for out-of-grid queries

- ``normalized``: When True, use grid positions normalized to [0,1]

Returns

- ``interpolation``: Result of interpolating ``coefficients`` at ``positions``, Tensor with shape ``[SHAPE, channels]``
