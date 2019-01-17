from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
from setuptools import find_packages
# from setuptools import Extension

exclude_dirs = ("configs", "tests", "scripts", "data", "outputs")

setup(
    name='shaper',
    version="0.0.4",
    author="Jiayuan Gu, Rui Chen, Maximilian Jaritz, Qinru Li, Yaokun Xu",
    url="https://github.com/haosulab/shaper",
    description="point clouds machine learning in pytorch",
    packages=find_packages(exclude=exclude_dirs),
)
