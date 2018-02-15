import tensorflow as tf
from tensorflow.python.framework import ops
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

spline_module = tf.load_op_library(os.path.join(dir_path,'splines.so'))
spline_grid = spline_module.spline_grid

@ops.RegisterGradient("SplineGrid")
def _(op, grad):
    pos = op.inputs[0]
    coeff = op.inputs[1]
    indices,values = spline_module.spline_grid_gradient(pos,grad,coeff.shape,order=op.get_attr('order'),dx=op.get_attr('dx'))
    return [None,tf.scatter_nd(indices, values, tf.shape(coeff))]
