import tensorflow as tf
from tensorflow.python.framework import ops
from functools import reduce
from operator import mul
import os
from distutils.sysconfig import get_config_var

def prod(iterable):
    return reduce(mul, iterable)

dir_path = os.path.dirname(os.path.realpath(__file__))
ext_suffix = get_config_var('EXT_SUFFIX')
name = [f for f in os.listdir(dir_path) if f.endswith(ext_suffix)][0]
library_file = os.path.join(dir_path, name)

spline_module = tf.load_op_library(library_file)

spline_grid = spline_module.spline_grid

try:
    @ops.RegisterGradient("SplineGrid")
    def _(op, grad):
        pos = op.inputs[0]
        coeff = op.inputs[1]
        indices,values = spline_module.spline_grid_coefficient_gradient(pos,grad,coeff.shape,order=op.get_attr('order'),dx=op.get_attr('dx'),periodic=op.get_attr('periodic'),debug=op.get_attr('debug'))
        pos_grad = spline_module.spline_grid_position_gradient(pos, coeff, grad,order=op.get_attr('order'),dx=op.get_attr('dx'),periodic=op.get_attr('periodic'),debug=op.get_attr('debug'))
        return [pos_grad,tf.scatter_nd(indices, values, tf.shape(coeff))]
except KeyError:
    pass




class SplineInterpolator:
    def __init__(self, C, order=[], periodic=[],extents=[]):
        self.C = C
        self.order = order
        self.periodic = periodic
        self.extents = extents

    def __call__(self, x):
        return spline_module.spline_grid(x,self.C,order=self.order,periodic=self.periodic)

    @property
    def dx(self):
        class D:
            def __getitem__(subself,dx):
                return lambda x: spline_grid(x,self.C,order=self.order,periodic=self.periodic,dx=dx)/\
                prod(self.extents[i]**dx[i] for i in range(len(dx)))

        return D()