# TensorSpline

## Introduction

This package provides support for B-spline tensor product interpolation and gradients of arbitrary dimensions, orders and differentials in TensorFlow.

To contribute to this project, talk to Thomas Gr�nli (thomas.gronli@gmail.com)

## Installation

Install from the command line with a C/C++ compiler suite installed

```console
$ pip install --upgrade git+https://bitbucket.org/ntnuultrasoundgroup/tensorspline.git
```


## Reference
The available functions have the following signatures  

### spline_grid

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



### spline_mapping

- ``mapping = spline_mapping(positions, values, weights, grid_shape, order=[], dx=[], periodic=[], fill_value=0)``

Required

- ``positions``: Locations of data points, Tensor or Tensor-like object with shape ``[SHAPE, d]``

- ``values``: Data point values, Tensor or Tensor-like object with shape ``[SHAPE, channels]``

- ``weights``: Data point weights, Tensor or Tensor-like object with shape ``[SHAPE, channels]``

- ``grid_shape``: Output grid size, list of ints length ``d``

Optional

- ``order``: Interpolation orders, list of ints length <= d, defaults to 3

- ``dx``: Differentiation orders, list of ints length <= d, defaults to 0

- ``periodic``: Grid periodicity, list of bools length <= d, defaults to False

- ``fill_value``: Fill value for regions without data support


Returns

- ``mapping``: Result of mapping ``values`` at ``positions`` onto ``grid_shape`` output grid, Tensor with shape ``[*grid_shape, channels]``

