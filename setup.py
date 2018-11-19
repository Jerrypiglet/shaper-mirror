from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
from setuptools import find_packages
# from Cython.Build import cythonize
# from setuptools import Extension

exclude_dirs = ("configs", "tests", "scripts", "data", "outputs")

setup(
    name='shaper',
    author="Rui Chen, Jiayuan Gu",
    version="0.0.0",
    packages=find_packages(exclude=(exclude_dirs)),
)
