#!/usr/bin/env bash
conda create -n shaper python=3.6
conda activate shaper
conda install pytorch torchvision cuda92 -c pytorch
# Basic tools
pip install cython yacs h5py
# Visualization
pip install matplotlib opencv-python
pip install tensorboardX
pip install open3d-python
# s2cnn
pip install cupy requests scipy pynvrtc