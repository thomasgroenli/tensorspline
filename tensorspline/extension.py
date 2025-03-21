import tensorflow as tf
from tensorflow.python.framework import ops
import ctypes

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
padding = spline_module.padding
b_spline = spline_module.b_spline

try:
    @ops.RegisterGradient("SplineGrid")
    def _(op, grad):
        pos = op.inputs[0]
        coeff = op.inputs[1]
        gradient = spline_mapping(pos,coeff,grad,order=op.get_attr('order'),dx=op.get_attr('dx'),periodic=op.get_attr('periodic'),debug=op.get_attr('debug'))
        return [None,gradient]
        
except KeyError:
    pass

try:
    @ops.RegisterGradient("Padding")
    def _(op, grad):
        tensor = op.inputs[0]
        pad_grad = spline_module.padding_gradient(grad,tensor,padding=op.get_attr('padding'),periodic=op.get_attr('periodic'))
        return [pad_grad]
        
except KeyError:
    pass


cdll = ctypes.CDLL(get_library_path())
cdll.set_launch_config.argtypes = [ctypes.c_int, ctypes.c_int]
cdll.set_launch_config.restype = None

cdll.cuda_enabled.argtypes = []
cdll.cuda_enabled.restype = ctypes.c_bool

set_launch_config = cdll.set_launch_config
cuda_enabled = cdll.cuda_enabled
