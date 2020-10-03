import tensorflow as tf
import numpy as np

from .extension import b_spline, padding


def generate_prefilter_kernel(p):
    x = tf.range(-4*p,4*p+1,dtype=tf.float32)
    y = b_spline(x,p,0)
    t = tf.signal.fft(tf.cast(y,np.complex128))
    p = tf.pad(tf.cast(tf.signal.ifft(1/t),tf.float32),[[0,2]],mode='CONSTANT')
    return p

def generate_bspline_kernel(p,dx):
    x = tf.cast(tf.range(-(p-1)//2-1,p//2+2)[::-1],tf.float32)
    return b_spline(x,p,dx)

def filter_ungrouped(C,kernels,ps,periodics):
    n_dim = len(C.shape)-1

    try:
        n_chan = C.shape[-1].value
    except AttributeError:
        n_chan = C.shape[-1]
    
    pads = sum(([p//2+1,p//2+1] for p in ps),[])
    C_tmp = [padding(C[...,i],pads,periodics) for i in range(n_chan)]
    
    permutation = np.roll(list(range(n_dim)),1)
    
    for i in range(len(C_tmp)):
        data = C_tmp[i]
        
        for kernel,p in reversed(list(zip(kernels,ps))):
            shape = tf.shape(data)
            new_shape = tf.concat([shape[:-1],[shape[-1]-2*(p//2+1)]],0)
            reshaped = tf.reshape(data,[tf.reduce_prod(shape[:-1]),shape[-1],1])
            kernel = kernel[:,None,None]
            conv = tf.nn.conv1d(reshaped,kernel,[1,1,1],'SAME')
            sliced = tf.slice(conv,[0,p//2+1,0],[tf.reduce_prod(new_shape[:-1]),new_shape[-1],1])
            data = tf.transpose(tf.reshape(sliced, new_shape),permutation)
 
        C_tmp[i] = data
            
    return tf.stack(C_tmp,axis=-1)

def filter_grouped(C,kernels,ps,periodics):
    n_dim = len(C.shape)-1

    pads = sum(([p//2+1,p//2+1] for p in ps),[])+[0,0]
    pers = list(periodics)+[False]
    C_tmp = padding(C,pads,pers)
    
    permutation = np.append(np.roll(list(range(n_dim)),1),n_dim)
        
    for kernel,p in reversed(list(zip(kernels,ps))):
        shape = tf.shape(C_tmp)
        new_shape = tf.concat([shape[:-2],[shape[-2]-2*(p//2+1)],[shape[-1]]],0)
        reshaped = tf.reshape(C_tmp,[tf.reduce_prod(shape[:-2]),shape[-2],-1])
        kernel = tf.tile(kernel[:,None,None],[1,1,shape[-1]])
        conv = tf.nn.conv1d(reshaped,kernel,[1,1,1],'SAME')
        sliced = tf.slice(conv,[0,p//2+1,0],[tf.reduce_prod(new_shape[:-2]),new_shape[-2],new_shape[-1]])
        data = tf.transpose(tf.reshape(sliced, new_shape),permutation)
        C_tmp = data
 
    return C_tmp

def filter(C,kernels,ps,periodics):
    try:
        return filter_grouped(C,kernels,ps,periodics)
    except tf.errors.UnimplementedError:
        return filter_ungrouped(C,kernels,ps,periodics)

def bspline_convolve(C,ps,periodics,dxs):
    kernels = [generate_bspline_kernel(p,dx) for p,dx in zip(ps,dxs)]
    dx_factor = tf.reduce_prod([(tf.cast(C.shape[i],tf.float32)-1+period)**dx 
                                    for i, (period,dx) in enumerate(zip(periodics,dxs))])
    return filter(C,kernels,ps,periodics)*dx_factor

def bspline_prefilter(C,ps,periodics):
    kernels = [generate_prefilter_kernel(p) for p in ps]
    return filter(C,kernels,ps,periodics)