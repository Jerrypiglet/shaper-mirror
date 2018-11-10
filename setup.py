from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
from setuptools import find_packages
# from Cython.Build import cythonize
# from setuptools import Extension


setup(
    name='shaper',
    version="0.0.0",
    packages=find_packages(exclude=("configs", "tests",)),
)
