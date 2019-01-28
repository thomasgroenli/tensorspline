import setuptools as core
import setuptools.extension as extension
import setuptools.command.build_ext as build_ext
import os

import pathlib
import platform

system = platform.system()

default_cuda_path = "/usr/local/cuda"

has_cuda = os.path.isdir(default_cuda_path) or 'CUDA_PATH' in os.environ

version = '1.10'
GPU_flag = {True: '-gpu', False: ''}
tf_req = 'tensorflow{0}=={1}'.format(GPU_flag[has_cuda], version)
      
def create_extension(distribution):

      distribution.fetch_build_eggs([tf_req])
      import tensorflow as tf
    
      if system == 'Windows':
            include_dirs = [str(tf.sysconfig.get_include())]
            library_dirs = [str(pathlib.Path(tf.sysconfig.get_lib()) / 'python')]
            libraries = ['pywrap_tensorflow_internal']
      
            macros = [("NOMINMAX",None),
                      ("COMPILER_MSVC",None),
                      ("USE_MULTITHREADING",None)
            ]
      
            sources = ['tensorspline/src/splines.cc', 'tensorspline/src/splinegrid_cpu.cc']

            extra_compile_args = []
      
            if has_cuda:
                  macros.append(("USE_GPU",None))
                  include_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'include'))
                  library_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'lib' / 'x64'))
                  libraries.extend(['cuda','cudart','nvrtc'])
                  sources.append('tensorspline/src/splinegrid_gpu.cc')

      elif system == 'Linux':    
            include_dirs = [str(tf.sysconfig.get_include())]
            library_dirs = [str(pathlib.Path(tf.sysconfig.get_lib()))]
            libraries = ['tensorflow_framework']

            macros = [("_GLIBCXX_USE_CXX11_ABI", "0"),
                      #("USE_MULTITHREADING",None) Multithreading in tensorflow broken on gcc>=5
            ]
      
            sources = ['tensorspline/src/splines.cc', 'tensorspline/src/splinegrid_cpu.cc']

            extra_compile_args = ['-std=c++11']
      
            if has_cuda:
                  macros.append(("USE_GPU",None))
                  include_dirs.append(str(pathlib.Path(default_cuda_path) / 'include'))
                  library_dirs.append(str(pathlib.Path(default_cuda_path) / 'lib64'))
                  libraries.extend(['cuda','cudart','nvrtc'])
                  sources.append('tensorspline/src/splinegrid_gpu.cc')

      else:
            raise Exception("Unknown target platform")
      

      ext = extension.Extension('tensorspline.tensorspline_library',
                                     define_macros = macros,
                                     include_dirs = include_dirs,
                                     libraries = libraries,
                                     library_dirs = library_dirs,
                                     sources = sources,
                                     extra_compile_args = extra_compile_args
            )
      ext._needs_stub = False
      return ext

class custom_build_ext(build_ext.build_ext):
      def run(self):
            self.extensions = [create_extension(self.distribution)]
            super().run()
            
      def get_export_symbols(self,ext):
            return ext.export_symbols
            

core.setup(name='TensorSpline',
      version='1.0',
      description='Tensorflow operation for nD spline interpolation',
      author='Thomas Grønli',
      author_email='thomas.gronli@gmail.com',
      packages=['tensorspline'],
      ext_modules=[extension.Extension('tensorspline.tensorspline_library',sources=[])],
      install_requires = [tf_req],
      cmdclass = {'build_ext': custom_build_ext}
     )
