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
| PointNet | 32 | 0.001 (Adam) | 0.0387 | 1354 | 0.0135 | 88.78 (0.21) | |
| PointNet++ (ssg) | 16 | 0.001 (Adam) | 0.0465 | 1638 | 0.0149 | 89.75 (0.30) | |
| DGCNN | 32 | 0.001 (Adam) | 0.2310 | 3995 | 0.1210 | 91.66 (0.16) | |
| DGCNN (plain) | 32 | 0.001 (Adam) | 0.1733 | 2543 | 0.0949 | 90.92 | w/o TNET, 1 trial |