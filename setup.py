import setuptools
from distutils.core import setup
import os
suffix = 'dll' if os.name == 'nt' else 'so' 


setup(name='TensorSpline',
      version='1.0',
      description='Tensorflow operation for nD spline interpolation',
      author='Thomas Grønli',
      author_email='thomas.gronli@gmail.com',
      packages=['tensorspline'],
      install_requires=['tensorflow-gpu'],
      package_data={'tensorspline': ['./lib/splines.'+suffix]}
     )
