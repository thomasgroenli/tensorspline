import tensorflow as tf
import numpy as np

from .extension import b_spline, padding


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
    x = tf.cast(tf.range(-(p-1)//2,p//2+1)[::-1],tf.float32)
    return b_spline(x,p,dx)


def bspline_convolve(C,ps,periodic,dxs):
    pads = sum(([p//2,p//2] for p in ps),[])+[0,0]
    pers = periodic+[False]
    C_tmp = padding(C,pads,pers)
    Ndim = len(C.shape)-1

    permutation = np.append(np.roll(list(range(Ndim)),1),Ndim)

    for kernel,p in reversed([(generate_1d_kernel(p,dx),p) for p,dx in zip(ps,dxs)]):
        shape = tf.shape(C_tmp)
        new_shape = tf.concat([shape[:-2],[shape[-2]-2*(p//2)],[shape[-1]]],0)
        
        C_tmp = tf.transpose(tf.reshape(tf.slice(tf.nn.conv2d(tf.reshape(C_tmp,
                                                                         [tf.reduce_prod(shape[:-2]),shape[-2],-1,1]),
                                                              tf.tile(kernel[:,None,None,None], 
                                                                      [1,shape[-1],1,1]),
                                                              [1,1,1,1],
                                                              'SAME'),
                                                 [0,p//2,0,0],
                                                 [tf.reduce_prod(new_shape[:-2]),new_shape[-2],shape[-1],1]),
                                        new_shape),
                             permutation)
    return C_tmp