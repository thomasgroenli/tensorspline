import tensorflow as tf
import numpy as np
import itertools as it
from .axes import Unitary, Legacy, transform_axes
from .extension import spline_module
from .kernels import bspline_prefilter, bspline_convolve

class SplineInterpolator:
    def __init__(self, C, axes=None, prefilter=False, fill_value=np.nan):
        self.ndim = len(C.shape)-1

        if axes is not None:
            self.axes = [axis if axis is not None else Unitary()
                         for axis in axes[:self.ndim]]
        else:
            self.axes = []
            
        while len(self.axes)<self.ndim:
            self.axes.append(Unitary())
        
        self.fill_value = fill_value
        
        if prefilter:
            self.C = bspline_prefilter(tf.cast(C,tf.float32), 
                                        [axis.order for axis in self.axes],
                                        [axis.period for axis in self.axes])
        else:
            self.C = C

            
    def transform(self, x):
        return transform_axes(tf.cast(x,tf.float32), self.axes)
        
    def __call__(self, x):
        if x is None:
            return bspline_convolve(self.C, ps=[axis.order for axis in self.axes],
                                            periodics=[bool(axis.period) for axis in self.axes],
                                            dxs=[0 for axis in self.axes])
        else:
            return spline_module.spline_grid(self.transform(x),
                                            self.C,
                                            order=[axis.order for axis in self.axes],
                                            periodic=[bool(axis.period) for axis in self.axes],
                                            fill_value=self.fill_value)

    @property
    def dx(self):
        class _:
            def __getitem__(_,dx):
                if isinstance(dx,int):
                    dx = (dx,)
                def _(x):
                    nonlocal dx
                    if x is None:
                        while len(dx)<self.ndim:
                            dx = dx+(0,)
                        res = bspline_convolve(self.C, ps=[axis.order for axis in self.axes],
                                                       periodics=[bool(axis.period) for axis in self.axes],
                                                       dxs=dx)
                    else:
                        res = spline_module.spline_grid(self.transform(x),
                                                        self.C,
                                                        order=[axis.order for axis in self.axes],
                                                        periodic=[bool(axis.period) for axis in self.axes],
                                                        fill_value=self.fill_value,
                                                        dx=dx)

                    return res/np.prod([axis.extent()**dx[i] for i,axis in enumerate(self.axes)])
                return _
        return _()


def LegacyInterpolator(C, order=[], periodic=[], extents=[]):
    axes = [Legacy(extent,order=p,period=period) for extent,p,period in it.zip_longest(extents,order,periodic,fillvalue=None)]
    return SplineInterpolator(C,axes,fill_value=0)