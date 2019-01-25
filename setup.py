import setuptools
import os

import tensorflow as tf

import pathlib
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import platform

system = platform.system()


if system == 'Windows':
      has_cuda = 'CUDA_PATH' in os.environ
      
      include_dirs = [str(tf.sysconfig.get_include())]
      library_dirs = [str(pathlib.Path(tf.sysconfig.get_lib()) / 'python')]
      libraries = ['pywrap_tensorflow_internal']
      
      macros = [("NOMINMAX",None),
                ("COMPILER_MSVC",None),
                ("USE_MULTITHREADING",None)
      ]
      
      sources = ['tensorspline/src/splines.cc', 'tensorspline/src/splinegrid_cpu.cc']

      if has_cuda:
            macros.append(("USE_GPU",None))
            include_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'include'))
            library_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'lib' / 'x64'))
            libraries.extend(['cuda','cudart','nvrtc'])
            sources.append('tensorspline/src/splinegrid_gpu.cc')

elif system == 'Linux':
      default_cuda_path = "/usr/local/cuda"
      #      has_cuda = 'CUDA_ROOT' in os.environ
      has_cuda = os.path.isdir(default_cuda_path)
      
      include_dirs = [str(tf.sysconfig.get_include())]
      library_dirs = [str(pathlib.Path(tf.sysconfig.get_lib()))]
      libraries = ['tensorflow_framework']

      macros = [("_GLIBCXX_USE_CXX11_ABI", "0"),
                #("USE_MULTITHREADING",None) Multithreading in tensorflow broken on gcc>=5
      ]
      
      sources = ['tensorspline/src/splines.cc', 'tensorspline/src/splinegrid_cpu.cc']
      if has_cuda:
            macros.append(("USE_GPU",None))
            include_dirs.append(str(pathlib.Path(default_cuda_path) / 'include'))
            library_dirs.append(str(pathlib.Path(default_cuda_path) / 'lib64'))
            libraries.extend(['cuda','cudart','nvrtc'])
            sources.append('tensorspline/src/splinegrid_gpu.cc')

else:
      raise Exception("Unknown target platform")
      

tensorspline = Extension('tensorspline.tensorspline_library',
                    define_macros = macros,
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    sources = sources,
                    extra_compile_args=['-std=c++11']
                    )

class custom_build_ext(build_ext):
      def get_export_symbols(self,ext):
            return ext.export_symbols

setuptools.setup(name='TensorSpline',
      version='1.0',
      description='Tensorflow operation for nD spline interpolation',
      author='Thomas Grønli',
      author_email='thomas.gronli@gmail.com',
      packages=['tensorspline'],
      ext_modules = [tensorspline],
      cmdclass = {'build_ext': custom_build_ext}
     )
