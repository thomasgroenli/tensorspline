import tensorflow as tf
import numpy as np

from .extension import b_spline, padding

def generate_prefilter_kernel(p):
    x = tf.range(-4*p,4*p+1,dtype=tf.float32)
    y = tf.cast(b_spline(x,p,0),tf.complex64)
    
    kernel = tf.signal.ifft(1/tf.signal.fft(y))
    return tf.math.real(kernel[2:]) if p > 0 else tf.constant([1],tf.float32)

def generate_bspline_kernel(p,dx,staggered):
    x = tf.cast(tf.range(-(p-1)//2-1-staggered,p//2+2)[::-1],tf.float32)+0.5*staggered
    return b_spline(x,p,dx)

def filter_ungrouped(C,kernels,periodics):
    n_dim = len(C.shape)-1

    try:
        n_chan = C.shape[-1].value
    except AttributeError:
        n_chan = C.shape[-1]
    
    C_tmp = [tf.cast(C[...,i],tf.float32) for i in range(n_chan)]
    
    permutation = np.roll(list(range(n_dim)),1)
    
    for i in range(len(C_tmp)):
        data = C_tmp[i]
        for kernel,periodic in reversed(list(zip(kernels,periodics))):
            shape = tf.shape(data)

            try:
                kernel_length = kernel.shape[0].value
            except AttributeError:
                kernel_length = kernel.shape[0]
            
            new_shape = tf.concat([shape[:-1],[shape[-1]-((kernel_length+1)%2)*(periodic!=1)]],0)

            reshaped = tf.reshape(data,[tf.reduce_prod(shape[:-1]),shape[-1],1])
            padded = padding(reshaped,[0,0,(kernel_length-1)//2,(kernel_length-1+(periodic==1))//2,0,0],[False,periodic,False])
            kernel = kernel[:,None,None]
            conv = tf.nn.conv1d(padded,kernel,[1,1,1],'VALID')
            data = tf.transpose(tf.reshape(conv, new_shape),permutation)
 
        C_tmp[i] = data
            
    return tf.stack(C_tmp,axis=-1)

def filter_grouped(C,kernels,periodics):
    n_dim = len(C.shape)-1
    
    permutation = np.append(np.roll(list(range(n_dim)),1),n_dim)
    
    C_tmp = tf.cast(C,tf.float32)
    
    for kernel,periodic in reversed(list(zip(kernels,periodics))):
        shape = tf.shape(C_tmp)
        
        try:
            kernel_length = kernel.shape[0].value
        except AttributeError:
            kernel_length = kernel.shape[0]

        new_shape = tf.concat([shape[:-2],[shape[-2]-((kernel_length+1)%2)*(periodic!=1)],[shape[-1]]],0)

        reshaped = tf.reshape(C_tmp,[tf.reduce_prod(shape[:-2]),shape[-2],-1])
        padded = padding(reshaped,[0,0,(kernel_length-1)//2,(kernel_length-1+(periodic==1))//2,0,0],[False,periodic,False])
        kernel = tf.tile(kernel[:,None,None],[1,1,shape[-1]])
        conv = tf.nn.conv1d(padded,kernel,[1,1,1],'VALID')
        data = tf.transpose(tf.reshape(conv, new_shape),permutation)
        
        C_tmp = data
 
    return C_tmp

def filter(C,kernels,periodics):
    try:
        return filter_grouped(C,kernels,periodics)
    except tf.errors.UnimplementedError:
        return filter_ungrouped(C,kernels,periodics)

def bspline_convolve(C,ps,periodics,dxs,staggers):
    kernels = [generate_bspline_kernel(p,dx,stagger) for p,dx,stagger in zip(ps,dxs,staggers)]
    dx_factor = tf.reduce_prod([(tf.cast(C.shape[i],tf.float32)-1+(period==1))**dx 
                                    for i, (period,dx) in enumerate(zip(periodics,dxs))])
    return filter(C,kernels,periodics)*dx_factor

def bspline_prefilter(C,ps,periodics):
    kernels = [generate_prefilter_kernel(p) for p in ps]
    return filter(C,kernels,periodics)

