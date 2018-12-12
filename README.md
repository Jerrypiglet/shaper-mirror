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
[setuptools](https://setuptools.readthedocs.io/en/latest/) is used to set up the python environment, 
so that the package is visible in PYTHONPATH.
```
# create anaconda environment
bash install.sh
# Remember to add develop so that all the modifications of python files could take effects.
python setup.py build develop
```
CUDA extensions are written to speed up calculations.
There are some [resources](#cuda-extension) to learn how to write cuda extensions for pytorch.
To run models including PointNet++, DGCNN, etc., source files should be compiled.
```
# take DGCNN for example
cd shaper/models/dgcnn_utils
python setup.py build_ext --inplace
```

## Getting Started

### Download datasets
Shaper currently supports several datasets, like ModelNet40 and ShapeNet.
Scripts to download data are provided in ``shaper``.
```
# take ModelNet40 for example
mkdir data
cd data
bash ../scripts/download_modelnet.sh
```

### Configuration
[YACS](https://pypi.org/project/yacs/), a simple experiment configuration system for research, 
is used to configure both training and testing.
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

## Contributing

### Pipeline
- Create a new branch for new features, bug fixes, or new projects.
- Add unittest for new codes
    - Add test for new models in-place
    - Add test for new operators or multiple functions in ``tests/``
- Pull request to master if the change is useful in general.

### Best practice
- Reuse the codes as much as possible.
    - Modular design
    - Inherit from existing classes
    - Add options instead of making a copy (if there are not too many options)
- Write unittest(pytest) for your codes in ``tests``.
- Use setup.py to build python packages and pytorch (cuda) extension.
- Create a new branch and a new folder for a new project.

## CUDA extension

``shaper/models/dgcnn_utils`` could be a good tutorial about how to write CUDA extension.
In general, ``setup.py`` will build extensions by compiling source files(".cpp", ".cu") within ``csrc``.

### Tutorials
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- https://pytorch.org/cppdocs
- https://github.com/pytorch/extension-cpp
- https://devblogs.nvidia.com/even-easier-introduction-cuda
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

## Troubleshooting

### Failure to setup ``shaper``
Run ``python setup.py build develop`` instead of ``python setup.py install develop``.

### Failure to find dynamic library(.so) after compiling source files.
There exist some errors in your source codes. For example, some functions are only declared.