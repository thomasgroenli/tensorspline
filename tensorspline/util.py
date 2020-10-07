import tensorflow as tf
import numpy as np

def grid(dims):
    return tf.stack(tf.meshgrid(*[np.linspace(0,1,x,dtype=np.float32) for x in dims],indexing='ij'),axis=-1)