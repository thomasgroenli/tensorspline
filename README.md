# TensorSpline

## Introduction

This package provides support for uniform B-spline tensor product interpolation and gradients of arbitrary dimensions, orders and differentials in TensorFlow.

To contribute to this project, talk to Thomas Grønli (thomas.gronli@gmail.com)

## Usage
### Installation
Install from the command line  

```console
$ pip install git+https://bitbucket.org/ntnuultrasoundgroup/tensorspline.git -v
```


### Reference
The available function has the following signature  

- ``interpolation = spline_grid(positions, coefficients, order=[], dx=[], periodic=[], fill_value=0)``

Required

- ``positions``: Query positions, Tensor or Tensor-like object with shape ``[SHAPE, d]``

- ``coefficients``: Coefficient grid, Tensor or Tensor-like object with shape ``[N1, N2, ..., Nd, channels]``

Optional

- ``order``: Interpolation orders, list of ints length <= d, defaults to 3

- ``dx``: Differentiation orders, list of ints length <= d, defaults to 0

- ``periodic``: Grid periodicity, list of bools length <= d, defaults to False

- ``fill_value``: Fill value for out-of-grid queries


Returns

- ``interpolation``: Result of interpolating ``coefficients`` at ``positions``, Tensor with shape ``[SHAPE, channels]``
