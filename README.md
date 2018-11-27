# Shaper
Shaper is the software system of Hao Su's Lab that implements state-of-the-art 3D Point Cloud algorithms, 
including PointNet and its variants. 
It is written in Python and powered by the Pytorch deep learning framework.

# Introduction
The goal of Shaper is to provide a high-quality, high-performance codebase for point cloud research. 
It is designed to be flexible in order to support rapid implementation and evaluation of novel research. 
Shaper includes implementations of the following point cloud algorithms:
- PointNet
- PointNet++
- DGCNN

# Model Zoo
## Baselines
| model | batch_size | lr | gpus | epoch | train time (s/iter) | train memory (GB) | accuracy | comments |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PointNet | 32 | 0.001 (Adam) | 1 | 250 | x | x | 88.7 (0.2) | 12 views ensemble |

# Installation
It is recommended to use (mini)conda to manage the environment.
```
bash install.sh  # create anaconda environment
python setup.py install develop
```

# Getting Started

# Best Practice
- Try to reuse the codes as many as possible
- Write unittest for your codes in tests/
- Make your code as concise and robust as possible
- Use setup.py to build python packages and pytorch (cuda) extension