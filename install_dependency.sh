#!/usr/bin/env bash
conda create -n shaper python=3.6
conda activate shaper
conda install pytorch torchvision cuda92 -c pytorch
# Basic tools
pip install cython yacs h5py scipy tqdm
# Point Cloud Related
pip install plyfile open3d-python
# Visualization
pip install matplotlib imageio
pip install tensorboardX PrettyTable
# s2cnn
pip install cupy requests pynvrtc