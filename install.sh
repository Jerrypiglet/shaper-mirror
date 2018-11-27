#!/usr/bin/env bash
# install dependency
conda create -n shaper python=3.6
conda activate shaper
conda install pytorch torchvision cuda92 -c pytorch
pip install -r requirements.txt
