import tensorflow as tf
from tensorflow.python.framework import ops
import os

from distutils.sysconfig import get_config_var


def get_library_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ext_suffix = get_config_var('EXT_SUFFIX')
    name = [f for f in os.listdir(dir_path) if f.endswith(ext_suffix)][0]
    return os.path.join(dir_path, name)

spline_module = tf.load_op_library(get_library_path())

spline_grid = spline_module.spline_grid
spline_mapping = spline_module.spline_mapping
b_spline = spline_module.b_spline

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