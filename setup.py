import setuptools
from distutils.core import setup, Extension
import os

import tensorflow as tf

import pathlib
from distutils.command import build_clib


has_cuda = 'CUDA_PATH' in os.environ

include_dirs = [str(tf.sysconfig.get_include())]
library_dirs = [str(pathlib.Path(tf.sysconfig.get_lib()) / 'python')]
libraries = ['pywrap_tensorflow_internal']

macros = [("NOMINMAX",None),("COMPILER_MSVC",None),("USE_GPU",None),("USE_MULTITHREADING",None)]
sources = ['src/splines.cc', 'src/splinegrid_cpu.cc']

if has_cuda:
      include_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'include'))
      library_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'lib' / 'x64'))
      libraries.extend(['cuda','cudart','nvrtc'])
      sources.append('src/splinegrid_gpu.cc')


libtensorspline = ('libtensorspline', {'sources': sources, 'include_dirs': include_dirs, 'macros': macros, 'libraries': libraries, 'library_dirs': library_dirs})


class custom_build_clib(build_clib.build_clib):
    def build_libraries(self, libs):
        for (lib_name, build_info) in libs: 
            self.compiler.link_shared_object(
                        self.compiler.compile(build_info.get('sources'),
                                            macros=build_info.get('macros'),
                                            include_dirs=build_info.get('include_dirs')),
                       'tensorspline/test.dll',
                        libraries=build_info.get('libraries'),
                        library_dirs=build_info.get('library_dirs'))



suffix = 'dll'
setup(name='TensorSpline',
      version='1.0',
      description='Tensorflow operation for nD spline interpolation',
      author='Thomas Grønli',
      author_email='thomas.gronli@gmail.com',
      packages=['tensorspline'],
      package_data={'tensorspline': ['./lib/test.'+suffix]},
      libraries = [libtensorspline],
      cmdclass={
        'build_clib': custom_build_clib,
        }
     )
