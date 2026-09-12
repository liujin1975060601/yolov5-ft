"""
    setup.py file for SWIG example
"""
import os
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import platform


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        build_ext.build_extensions(self)

ext_modules = [
    Extension(name='poly_cpu',
              sources=['poly_cpu_func.cpp', 'poly_cpu.pyx'],
              language='c++',
              # this syntax is specific to this build system
              # we're only going to use certain compiler args with nvcc and not with
              # gcc the implementation of this trick is in customize_compiler() below
            #   extra_compile_args=['-fopenmp' if platform.system() == 'Linux' else ''],
            #   extra_link_args=['-fopenmp' if platform.system() == 'Linux' else ''],
              include_dirs = [numpy_include],
              )
]

setup(
      ext_modules=cythonize(ext_modules, language_level=3),
      # inject our custom trigger
      cmdclass={'build_ext': custom_build_ext},
)
