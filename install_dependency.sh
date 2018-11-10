#!/usr/bin/env bash
conda create -n pc python=3.6
conda activate pc
conda install pytorch torchvision cuda92 -c pytorch
pip install cython yacs
pip install matplotlib opencv-python
# 3D visualization
pip install open3d-python