# Shaper Model Zoo and Baselines

## Environment
- Ubuntu 18.04.1 LTS
- NVIDIA 1080Ti
- CUDA 9.2
- CUDNN 7.1
- PyTorch 0.41

## Classification Baselines
Without specification, the experiment setting is to train one model for 250 epochs,
and test it with multi-view(12) ensemble 5 times,
which might alleviate the randomness caused by the size of dataset.
By default, only one gpu is used for comparision.

| model | batch_size | lr | train time (s/iter) | train memory (GB) | test time (s/iter) | accuracy | comments |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PointNet | 32 | 0.001 (Adam) | 0.0378 | 1265 | 0.0135 | 89.04 (0.13) | |
| PointNet++ (ssg) | 16 | 0.001 (Adam) | 0.0479 | 1383 | 0.0155 | 90.18 (0.17) | |
| DGCNN | 32 | 0.001 (Adam) | 0.2287 | 3683 | 0.1223 | 91.61 (0.24) | |