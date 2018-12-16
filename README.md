# Shaper
Shaper is the software system of Hao Su's Lab that implements state-of-the-art 3D Point Cloud algorithms, 
including PointNet and its variants. 
It is written in Python and powered by the Pytorch deep learning framework.

## Introduction
The goal of Shaper is to provide a high-quality, high-performance codebase for point cloud research. 
It is designed to be flexible in order to support rapid implementation and evaluation of novel research. 
Shaper includes implementations of the following point cloud algorithms:
- PointNet
- PointNet++
- DGCNN

## Model Zoo
Please check [Model Zoo](MODEL_ZOO.md) for benchmark results.

## Installation
It is recommended to use (mini)conda to manage the environment.
```
bash install.sh  # create anaconda environment
python setup.py install develop (--user)
```

## Getting Started

### Download datasets
Shaper currently supports several datasets, like ModelNet40 and ShapeNet.
Scripts to download data are provided in ``shaper``.
```
mkdir data
cd data
bash ../scripts/download_modelnet.sh
```

### Configuration
[YACS](https://pypi.org/project/yacs/), a simple experiment configuration system for research, is used to configure both training and testing.
It is a library developed by Facebook Research and used in projects like Detectron.

### Train model
```
python tools/train_net.py --cfg=configs/baselines/pointnet_cls.yaml
```
The training logs, model weights, and tensorboard events will be saved to a directory provided in yaml.
Tensorboard is supported to monitor the training status.

### Test model
```
python tools/test_net.py --cfg=configs/baselines/pointnet_cls.yaml
```

### Unittest
[pytest](https://docs.pytest.org/en/latest/) is recommended for unittest, which could be installed by ``pip``.
It will automatically run all the functions and python files starting with **test**.
```
cd tests
# run all the files starting with "test"
pytest -s
# run all the functions starting with "test" within "test_functional.py"
pytest -s test_functional.py
```

## Best Practice
- Reuse the codes.
    - Modular design
    - Inherit from existing classes
    - Add options instead of making a copy (if there are not too many options) 
- Write unittest(pytest) for your codes in ``tests``.
- Use setup.py to build python packages and pytorch (cuda) extension.
- Create a new branch and a new folder for a new project.