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
    x = tf.cast(tf.range(-(p-1)//2-1,p//2+2)[::-1],tf.float32)
    return b_spline(x,p,dx)

def bspline_convolve_ungrouped(C,ps,periodics,dxs):
    n_dim = len(C.shape)-1

    try:
        n_chan = C.shape[-1].value
    except AttributeError:
        n_chan = C.shape[-1]
    
    pads = sum(([p//2+1,p//2+1] for p in ps),[])
    C_tmp = [padding(C[...,i],pads,periodics) for i in range(n_chan)]
    
    permutation = np.roll(list(range(n_dim)),1)
    
    dx_factor = 1
    
    for i in range(len(C_tmp)):
        data = C_tmp[i]
        
        for p,periodic,dx in reversed(list(zip(ps,periodics,dxs))):
            shape = tf.shape(data)
            new_shape = tf.concat([shape[:-1],[shape[-1]-2*(p//2+1)]],0)

            reshaped = tf.reshape(data,[tf.reduce_prod(shape[:-1]),shape[-1],1])
            kernel = generate_1d_kernel(p,dx)[:,None,None]
            conv = tf.nn.conv1d(reshaped,kernel,[1,1,1],'SAME')
            sliced = tf.slice(conv,[0,p//2+1,0],[tf.reduce_prod(new_shape[:-1]),new_shape[-1],1])
            data = tf.transpose(tf.reshape(sliced, new_shape),permutation)
            if i==0:
                dx_factor *= (tf.cast(new_shape[-1],tf.float32)-1+periodic)**dx
 
        C_tmp[i] = data
            
    return tf.stack(C_tmp,axis=-1)*dx_factor


def bspline_convolve_grouped(C,ps,periodics,dxs):
    n_dim = len(C.shape)-1

    try:
        n_chan = C.shape[-1].value
    except AttributeError:
        n_chan = C.shape[-1]
    
    pads = sum(([p//2+1,p//2+1] for p in ps),[])+[0,0]
    pers = list(periodics)+[False]
    C_tmp = padding(C,pads,periodics)
    
    permutation = np.append(np.roll(list(range(n_dim)),1),n_dim)
        
    for p,periodic,dx in reversed(list(zip(ps,periodics,dxs))):
        shape = tf.shape(C_tmp)
        new_shape = tf.concat([shape[:-2],[shape[-2]-2*(p//2+1)],[shape[-1]]],0)
        reshaped = tf.reshape(C_tmp,[tf.reduce_prod(shape[:-2]),shape[-2],-1])
        kernel = tf.tile(generate_1d_kernel(p,dx)[:,None,None],[1,1,shape[-1]])
        conv = tf.nn.conv1d(reshaped,kernel,[1,1,1],'SAME')
        sliced = tf.slice(conv,[0,p//2+1,0],[tf.reduce_prod(new_shape[:-2]),new_shape[-2],new_shape[-1]])
        data = tf.transpose(tf.reshape(sliced, new_shape),permutation)
        C_tmp = data*(tf.cast(new_shape[-2],tf.float32)-1+periodic)**dx
 
    return C_tmp

def bspline_convolve(C,ps,periodics,dxs):
    try:
        return bspline_convolve_grouped(C,ps,periodics,dxs)
    except tf.errors.UnimplementedError:
        return bspline_convolve_ungrouped(C,ps,periodics,dxs)