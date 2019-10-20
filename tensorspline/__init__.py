import tensorflow as tf
from tensorflow.python.framework import ops
from functools import reduce
from operator import mul
import os
import numpy as np

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


def generate_1d_kernel(p,dx):
    positions = 0.5*np.ones([1,1],dtype=np.float32)
    gradients = np.ones([1,1],dtype=np.float32)
    grid = spline_module.spline_grid_coefficient_gradient(coeff_shape=[2,1],
                                                             positions=positions,
                                                             gradients=gradients,
                                                             order=[p],dx=[dx])
    return grid[1]


def bspline_convolve(C,ps,dxs):
    C_tmp = C
    Ndim = len(ps)
    permutation = np.append(np.roll(list(range(Ndim)),1),Ndim)

    for kernel, p in reversed([(generate_1d_kernel(p,dx),p) for p,dx in zip(ps,dxs)]):
        shape = tf.shape(C_tmp)
        new_shape = tf.concat([shape[:-2],[shape[-2]-p],[shape[-1]]],0)
        C_tmp = tf.transpose(tf.reshape(tf.nn.conv1d(tf.reshape(C_tmp,[tf.reduce_prod(shape[:-2]),shape[-2],-1]),tf.tile(kernel[:,None,None],[1,1,shape[-1]]),1,'VALID'),new_shape),permutation)
    return C_tmp


def generate_prefilter_kernel(p):
    x = tf.range(-64,65,dtype=tf.float32)
    y = spline_module.b_spline(x,p)
    t = tf.signal.fft(tf.cast(y,np.complex128))
    p = tf.pad(tf.cast(tf.signal.ifft(1/t),tf.float32),[[0,2]],mode='CONSTANT')
    return p

def bspline_prefilter(C,ps):
    C_tmp = C
    Ndim = len(ps)
    permutation = np.append(np.roll(list(range(Ndim)),1),Ndim)

    shape = tf.shape(C_tmp)
    
    for kernel in reversed([generate_prefilter_kernel(p) for p in ps]):    
        C_tmp = tf.transpose(tf.reshape(tf.nn.convolution(tf.reshape(C_tmp,[tf.reduce_prod(shape[:-2]),shape[-2],-1]),tf.tile(kernel[:,None,None],[1,1,shape[-1]]),1,'SAME'),shape),permutation)
    
    return C_tmp

class SplineInterpolator:
    def __init__(self, C, order=[], periodic=[],extents=[]):
        self.C = C
        self.order = order
        self.periodic = periodic
        self.extents = extents

    def __call__(self, x):
        if x is None:
            return bspline_convolve(self.C,self.order,[0]*len(self.order))
        return spline_module.spline_grid(x,self.C,order=self.order,periodic=self.periodic)

    @property
    def dx(self):
        class D:
            def __getitem__(subself,dx):
                def dummy(x):
                    if x is None:
                        res = bspline_convolve(self.C,self.order,dx)*tf.reduce_prod(tf.cast(tf.shape(self.C)[:-1],tf.float32)**dx)
                    else:
                        res = spline_module.spline_grid(x,self.C,order=self.order,periodic=self.periodic,dx=dx)
                    return res/np.prod([self.extents[i]**dx[i] for i in range(len(dx))])
                return dummy
        return D()

