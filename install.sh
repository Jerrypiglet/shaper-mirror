#!/usr/bin/env bash
# install dependency
conda create -n shaper python=3.6
source activate shaper
# install pytorch 1.0 with cuda 9.0
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
