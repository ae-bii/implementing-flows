from setuptools import setup, Extension
import os


os.environ["CC"] = "gcc-9"

ext = Extension(
      'c_extension',
      sources = ['c_extension.c'],
      extra_compile_args=['-fopenmp'],
      extra_link_args=['-lgomp'])

setup(name='c_extension',
       version='1.0',
       description='This is a testing package',
       ext_modules=[ext])