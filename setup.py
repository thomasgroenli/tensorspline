import setuptools as core
import setuptools.extension as extension
import setuptools.command.build_ext as build_ext
import os

import logging
import pathlib
import platform

logging.basicConfig(level=logging.INFO)


def create_extension(distribution, cuda_path=None):

      tf = __import__("tensorflow")

      system = platform.system()

      if system == 'Windows':
            inc_path = pathlib.Path(tf.sysconfig.get_include())
            
            import distutils

            distutils.dir_util.copy_tree(inc_path, 'build/include')
            try:
                  os.rename('build/include/tensorflow_core/', 'build/include/tensorflow')
            except:
                  pass
            include_dirs = ["build/include"]

            distutils.file_util.copy_file(
                  pathlib.Path(tf.sysconfig.get_lib()) / 'python' / '_pywrap_tensorflow_internal.lib',
                  'build/tensorflow.lib'
            )
            library_dirs = ['build']
            libraries = ['tensorflow']
      
            macros = [("NOMINMAX",None),
                      ("COMPILER_MSVC",None),
                      ("USE_MULTITHREAD",None)
            ]
      
            sources = ['tensorspline/src/splines.cc', 
            'tensorspline/src/padding.cc', 'tensorspline/src/padding_cpu.cc',
             'tensorspline/src/splinegrid_cpu.cc', 'tensorspline/src/splinemapping_cpu.cc']

            extra_compile_args = []
            extra_link_args = []
      
            if cuda_path is not None:
                  macros.append(("USE_GPU",None))
                  include_dirs.append(str(pathlib.Path(cuda_path) / 'include'))
                  library_dirs.append(str(pathlib.Path(cuda_path) / 'lib' / 'x64'))
                  libraries.extend(['cuda','cudart','nvrtc'])
                  sources.extend(['tensorspline/src/splinegrid_gpu.cc', 'tensorspline/src/splinemapping_gpu.cc', 'tensorspline/src/padding_gpu.cc'])

      elif system in ['Darwin', 'Linux']:
            inc_path = pathlib.Path(tf.sysconfig.get_include())

            import distutils

            distutils.dir_util.copy_tree(inc_path, 'build/include')
            try:
                  os.rename('build/include/tensorflow_core/', 'build/include/tensorflow')
            except:
                  pass
            include_dirs = ["build/include"]

            distutils.file_util.copy_file(
                  pathlib.Path(tf.sysconfig.get_lib()) / 'python' / '_pywrap_tensorflow_internal.so',
                  'build/libtensorflow.so'
            )
            library_dirs = ['build']
            libraries = ['tensorflow']

            macros = [("_GLIBCXX_USE_CXX11_ABI", "0"),
                      ("USE_MULTITHREAD",None)
            ]
      
            sources = ['tensorspline/src/splines.cc', 
            'tensorspline/src/padding.cc', 'tensorspline/src/padding_cpu.cc',
             'tensorspline/src/splinegrid_cpu.cc', 'tensorspline/src/splinemapping_cpu.cc']

            extra_compile_args = ['-std=c++11']
            extra_link_args = ['-stdlib=libc++']
      
            if cuda_path is not None:
                  macros.append(("USE_GPU",None))
                  include_dirs.append(str(pathlib.Path(cuda_path) / 'include'))
                  library_dirs.append(str(pathlib.Path(cuda_path) / 'lib64'))
                  libraries.extend(['cuda','cudart','nvrtc'])
                  sources.extend(['tensorspline/src/splinegrid_gpu.cc', 'tensorspline/src/splinemapping_gpu.cc', 'tensorspline/src/padding_gpu.cc'])

      else:
            raise Exception("Unknown target platform")
      

      ext = extension.Extension('tensorspline.tensorspline_library',
                                     define_macros = macros,
                                     include_dirs = include_dirs,
                                     libraries = libraries,
                                     library_dirs = library_dirs,
                                     sources = sources,
                                     extra_compile_args = extra_compile_args,
                                     extra_link_args = extra_link_args
                                
            )
      ext._needs_stub = False
      return ext

class custom_build_ext(build_ext.build_ext):

      user_options = build_ext.build_ext.user_options+[
        ('cuda=', None, 'Specify the CUDA installation path.'),
      ]

      def initialize_options(self):
            super().initialize_options()
            self.cuda = None
      
      def finalize_options(self):
            if self.cuda == 'auto':
                  if 'CUDA_PATH' in os.environ:
                        self.cuda = os.environ['CUDA_PATH']
                  else:
                        self.cuda = None
            super().finalize_options()

      def run(self):
            self.extensions = [create_extension(self.distribution, self.cuda)]
            super().run()
            
      def get_export_symbols(self,ext):
            return ext.export_symbols+['set_launch_config']
            

core.setup(name='tensorspline',
      description='Tensorflow operation for nD spline interpolation',
      version='1.1.1',
      author='Thomas Grønli',
      author_email='thomas.gronli@gmail.com',
      packages=['tensorspline'],
      ext_modules=[extension.Extension('tensorspline.tensorspline_library',sources=[])],
      cmdclass = {'build_ext': custom_build_ext}
     )
