import tensorflow as tf
from tensorflow.python.framework import ops
import os
import math
dir_path = os.path.dirname(os.path.realpath(__file__))

spline_module = tf.load_op_library(os.path.join(dir_path,'splines.so'))
spline_grid = spline_module.spline_grid

@ops.RegisterGradient("SplineGrid")
def _(op, grad):
    pos = op.inputs[0]
    coeff = op.inputs[1]
    indices,values = spline_module.spline_grid_gradient(pos,grad,coeff.shape,order=op.get_attr('order'),dx=op.get_attr('dx'))
    return [None,tf.scatter_nd(indices, values, tf.shape(coeff))]


# Spline grid composed with tensorflow ops
def spline_grid_tf(x,C,n=3):
    dims = len(C.shape)-1
    M = tf.transpose(tf.meshgrid(*([tf.range(0,4,dtype=tf.int32)]*dims)))
    fac = x*(tf.cast(tf.shape(C)[:-1],tf.float32)-3)
    slc = (slice(None),)+dims*(None,)+(...,)
    indices = tf.cast(fac,tf.int32)
    coeff = tf.gather_nd(C, M[None]+indices[slc])
    offsets = (tf.mod(fac,1)[slc]+1-tf.cast(M,tf.float32))
    sigmasq = (n+1)/12 
    f = tf.reduce_prod(1/math.sqrt(2*math.pi*sigmasq)*tf.exp(-0.5*offsets*offsets/sigmasq),axis=-1)
    return tf.reduce_sum(f[...,None]*coeff,axis=tf.range(1,dims+1))
