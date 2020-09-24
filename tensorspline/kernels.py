import tensorflow as tf
import numpy as np

from .extension import b_spline


def generate_prefilter_kernel(p):
    x = tf.range(-64,65,dtype=tf.float32)
    y = b_spline(x,p)
    t = tf.signal.fft(tf.cast(y,np.complex128))
    p = tf.pad(tf.cast(tf.signal.ifft(1/t),tf.float32),[[0,2]],mode='CONSTANT')
    return p

def bspline_prefilter(C,ps):
    C_tmp = C
    
    Ndim = len(ps)

    permutation = np.append(np.roll(list(range(Ndim)),1),Ndim)

    for kernel in reversed([generate_prefilter_kernel(p) for p in ps]):
        shape = tf.shape(C_tmp)
        new_shape = tf.concat([shape[:-2],[shape[-2]],[shape[-1]]],0)   
        C_tmp = tf.transpose(tf.reshape(tf.nn.convolution(tf.reshape(C_tmp,[tf.reduce_prod(shape[:-2]),shape[-2],-1]),tf.tile(kernel[:,None,None],[1,1,shape[-1]]),1,'SAME'),new_shape),permutation)
    
    return C_tmp

def generate_1d_kernel(p,dx):
    x = tf.cast(tf.range(-(p-1)//2-1,p//2+2)[::-1],tf.float32)
    return b_spline(x,p,dx)

def bspline_convolve(C,ps,dxs):
    C_tmp = C

    Ndim = len(ps)

    permutation = np.append(np.roll(list(range(Ndim)),1),Ndim)

    for kernel in reversed([generate_1d_kernel(p,dx) for p,dx in zip(ps,dxs)]):
        shape = tf.shape(C_tmp)
        new_shape = tf.concat([shape[:-2],[shape[-2]],[shape[-1]]],0)
        C_tmp = tf.transpose(tf.reshape(tf.nn.conv1d(tf.reshape(C_tmp,[tf.reduce_prod(shape[:-2]),shape[-2],-1]),tf.tile(kernel[:,None,None],[1,1,shape[-1]]),1,'SAME'),new_shape),permutation)
    return C_tmp
