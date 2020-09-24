import tensorflow as tf
import numpy as np

class Axis:
    def __init__(self, order=1, period=0):
        self.order = order
        self.period = period

class Unitary(Axis):
    def extent(self):
        return 1
        
class Uniform(Axis):
    def __init__(self,a,b,**kwargs):
        assert b>=a
        self.a = a
        self.b = b
        super().__init__(**kwargs)
        
    def extent(self):
        return self.b-self.a
    
class Nonuniform(Axis):
    def __init__(self, x, **kwargs):
        assert np.all(np.diff(x)>=0)
        self.x = x
        super().__init__(**kwargs)
        
    def extent(self):
        return self.x[-1]-self.x[0]


def transform_axes(x, axes):
    if all(isinstance(axis, Unitary) for axis in axes):
        return x
    
    return tf.stack([transform_axis(x[..., i], axis) for i, axis in enumerate(axes)],axis=-1)
    
def transform_axis(x, axis):
    if isinstance(axis, Unitary):
        return x
    
    elif isinstance(axis, Uniform):
        return (x-axis.a)/(axis.b-axis.a)
    
    elif isinstance(axis, Nonuniform):
        k = list(axis.x)
        
        if axis.period:
            k = [ki%axis.period for ki in k]
            k = [k[-1]-axis.period]+k+[k[0]+axis.period]

            x = x % axis.period

        k = tf.constant(k, x.dtype)
        idx = tf.reshape(tf.searchsorted(k[None],tf.reshape(x,[-1])),x.shape)
        if not axis.period:
            idx = tf.clip_by_value(idx,0,len(k)-1)

        a,b = tf.gather(k,idx-1), tf.gather(k,idx)
        return ((x-a)/(b-a)+(tf.cast(idx,x.dtype)-1)-bool(axis.period))/(len(k)-1-bool(axis.period))