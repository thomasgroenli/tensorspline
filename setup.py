import setuptools as core
import setuptools.extension as extension
import setuptools.command.build_ext as build_ext
import os

import pathlib
import platform

system = platform.system()


has_cuda = "CUDA_PATH" in os.environ


GPU_flag = {True: '-gpu', False: ''}
tf_req = 'tensorflow{0}==2.0'.format(GPU_flag[has_cuda])

def create_extension(distribution):

      tf = __import__("tensorflow")
    
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
      
            sources = ['tensorspline/src/splines.cc', 'tensorspline/src/splinegrid_cpu.cc', 'tensorspline/src/splinemapping_cpu.cc']

            extra_compile_args = []
      
            if has_cuda:
                  macros.append(("USE_GPU",None))
                  include_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'include'))
                  library_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'lib' / 'x64'))
                  libraries.extend(['cuda','cudart','nvrtc'])
                  sources.append('tensorspline/src/splinegrid_gpu.cc')

      elif system == 'Linux':    
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
      
            sources = ['tensorspline/src/splines.cc', 'tensorspline/src/splinegrid_cpu.cc', 'tensorspline/src/splinemapping_cpu.cc']

            extra_compile_args = ['-std=c++11']
      
            if has_cuda:
                  macros.append(("USE_GPU",None))
                  include_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'include'))
                  library_dirs.append(str(pathlib.Path(os.environ['CUDA_PATH']) / 'lib64'))
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
            

core.setup(name='tensorspline',
      description='Tensorflow operation for nD spline interpolation',
      author='Thomas Grønli',
      author_email='thomas.gronli@gmail.com',
      packages=['tensorspline'],
      ext_modules=[extension.Extension('tensorspline.tensorspline_library',sources=[])],
      setup_requires = [tf_req],
      install_requires = [tf_req],
      cmdclass = {'build_ext': custom_build_ext}
     )
