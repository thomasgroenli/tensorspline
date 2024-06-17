import warnings
import unittest
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import numpy as np
import tensorspline as ts
import tensorflow as tf

import logging
logging.captureWarnings(True)

def allclose(A,B,rtol=1e-3,atol=1e-8):
    return np.all(np.abs(A-B)<rtol*np.sqrt(np.mean(np.square(B)))+atol)

class TestPadding(unittest.TestCase):
    device = '/cpu:0'

    def setUp(self):
        self.A = np.random.rand(10,10)

    def test_reflect(self):
        with tf.device(self.__class__.device):
            padded = ts.extension.padding(self.A, [2,2,2,2], [-1,-1])
        
        #Left edge
        self.assertTrue(allclose(padded[:2,2:-2][::-1], self.A[1:3]))    
        #Right edge
        self.assertTrue(allclose(padded[-2:,2:-2][::-1], self.A[-3:-1]))
        # Top edge
        self.assertTrue(allclose(padded[2:-2,:2][:,::-1], self.A[:,1:3]))
        # Bottom edge
        self.assertTrue(allclose(padded[2:-2,-2:][:,::-1], self.A[:,-3:-1]))

    def test_constant(self):
        with tf.device(self.__class__.device):
            padded = ts.extension.padding(self.A, [2,2,2,2], [0,0])
        
        #Left edge
        self.assertTrue(allclose(padded[:2,2:-2], self.A[:1]))
        #Right edge
        self.assertTrue(allclose(padded[-2:,2:-2], self.A[-1:]))
        # Top edge
        self.assertTrue(allclose(padded[2:-2,:2], self.A[:,:1]))
        # Bottom edge
        self.assertTrue(allclose(padded[2:-2,-2:], self.A[:,-1:]))

    def test_periodic(self):
        with tf.device(self.__class__.device):
            padded = ts.extension.padding(self.A, [2,2,2,2], [1,1])
        
        #Left edge
        self.assertTrue(allclose(padded[:2,2:-2], self.A[-2:]))
        #Right edge
        self.assertTrue(allclose(padded[-2:,2:-2], self.A[:2]))
        # Top edge
        self.assertTrue(allclose(padded[2:-2,:2], self.A[:,-2:]))
        # Bottom edge
        self.assertTrue(allclose(padded[2:-2,-2:], self.A[:,:2]))

    def test_reflect_gradient(self):
        @tf.function
        def test_pad(A):
            return ts.extension.padding(A,[2,2,2,2],[-1,-1])
        
        with tf.device(self.__class__.device):
            A = tf.constant(self.A, dtype=tf.float32)
            grads = tuple(tf.test.compute_gradient(test_pad,[A]))

        self.assertTrue(allclose(grads[0][0],grads[1][0]))

    def test_constant_gradient(self):
        @tf.function
        def test_pad(A):
            return ts.extension.padding(A,[2,2,2,2],[0,0])
        
        with tf.device(self.__class__.device):
            A = tf.constant(self.A, dtype=tf.float32)
            grads = tuple(tf.test.compute_gradient(test_pad,[A]))

        self.assertTrue(allclose(grads[0][0],grads[1][0]))

    def test_periodic_gradient(self):
        @tf.function
        def test_pad(A):
            return ts.extension.padding(A,[2,2,2,2],[1,1])
        
        with tf.device(self.__class__.device):
            A = tf.constant(self.A, dtype=tf.float32)
            grads = tuple(tf.test.compute_gradient(test_pad,[A]))

        self.assertTrue(allclose(grads[0][0],grads[1][0]))         


class TestPrefilter(unittest.TestCase):
    device = '/cpu:0'

    def setUp(self):
        self.n = 11
        self.A = np.random.rand(self.n,1)
        self.x = np.linspace(0,1,self.n)[:,None]


    def test_prefilter(self):
        with tf.device(self.__class__.device):
            for order in range(5):
                for periodic in [-1,1]:
                    with self.subTest(i=order,periodic=periodic):
                        x_A = self.x
                        if periodic==1:
                            x_A = x_A*(self.n-1)/self.n
                        A_pre = ts.bspline_prefilter(self.A,[order],[periodic])
                        A_int = ts.spline_grid(x_A, A_pre, order=[order], periodic=[periodic])
                        self.assertTrue(allclose(A_int,self.A))


class TestConvolution(unittest.TestCase):
    device = '/cpu:0'

    def setUp(self):
        self.n = 11
        self.A = np.random.rand(self.n,1)
        self.x = np.linspace(0,1,self.n)[:,None]

    def test_convolution(self):
        with tf.device(self.__class__.device):
            for order in range(5):
                for periodic in [-1,0,1]:
                    for stagger in [False, True]:
                        for dx in range(order):
                            with self.subTest(order=order,periodic=periodic,stagger=stagger,dx=dx):
                                x_A = self.x*((self.n-1)/self.n if periodic==1 else 1)
                                if stagger and periodic==1:
                                    x_A = x_A+0.5/self.n
                                elif stagger:
                                    x_A = x_A[:-1]+0.5/(self.n-1)
                                A_conv = ts.bspline_convolve(self.A,[order],[periodic],[dx],[stagger])
                                A_int = ts.spline_grid(x_A, self.A, order=[order], periodic=[periodic], dx=[dx])
                                self.assertTrue(allclose(A_int,A_conv))


class TestSplineGrid(unittest.TestCase):
    device = '/cpu:0'
    
    def test_constant(self):
        with tf.device(self.__class__.device):
            A = np.zeros([11,1])
            x = np.linspace(0,1,101)[:,None]
            for order in range(5):
                for periodic in [-1,0,1]:
                    for dx in range(3):
                        with self.subTest(order=order,period=periodic,dx=dx):
                            A_int = ts.spline_grid(x,A,order=[order],periodic=[periodic],dx=[dx])
                            self.assertTrue(allclose(A_int,0))

    def test_periodic(self):
        with tf.device(self.__class__.device):
            A = np.random.rand(10,1)
            x = np.linspace(0,1,10)[:,None]*9/10
            for order in range(5):
                for dx in range(3):
                    for offset in range(-5,6):
                        if order == dx == 1:
                            continue
                        with self.subTest(order=order,dx=dx,offset=offset):
                            A_int = ts.spline_grid(x,A,order=[order],periodic=[1],dx=[dx])
                            A_int_offset = ts.spline_grid(x+offset,A,order=[order],periodic=[1],dx=[dx])
                            self.assertTrue(allclose(A_int,A_int_offset))

    def test_gradient(self):
        with tf.device(self.__class__.device):
            C = tf.constant(np.random.rand(11,1),tf.float32)
            x = tf.constant(np.linspace(0,1,101)[:,None],tf.float32)

            for order in range(5):
                for periodic in [-1,0,1]:
                    for dx in range(3):
                        with self.subTest(order=order,periodic=periodic,dx=dx):
                            @tf.function
                            def test_spline(C):
                                return ts.spline_grid(x,C,order=[order],periodic=[periodic],dx=[dx])

                            grads = tuple(tf.test.compute_gradient(test_spline,[C]))
                            self.assertTrue(allclose(grads[0][0],grads[1][0]))


class TestPaddingGPU(TestPadding):
    device = '/gpu:0'

class TestPrefilterGPU(TestPrefilter):
    device = '/gpu:0'

class TestConvolutionGPU(TestConvolution):
    device = '/gpu:0'

class TestSplineGridGPU(TestSplineGrid):
    device = '/gpu:0'


if __name__ == '__main__':
    unittest.main()
