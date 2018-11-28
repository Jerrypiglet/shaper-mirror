# Shaper Model Zoo and Baselines

## Introduction
The common setting for experiments is to train one model and test it with multi-view ensemble 5 times,
which might alleviate the randomness caused by the size of dataset.
By default, only one gpu is used for comparision.

## Classification Baselines
| model | batch_size | lr | gpus | epoch | train time (s/iter) | train memory (GB) | accuracy | comments |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PointNet | 32 | 0.001 (Adam) | 1 | 250 | 0.0387 | 1354 | 88.78 (0.21) | 12 views ensemble |
| DGCNN | 32 | 0.001 (Adam) | 1 | 250 | 0.2587| 4795 | 91.51 (0.41) | 12 views ensemble |