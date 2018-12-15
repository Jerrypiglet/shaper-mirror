#!/usr/bin/env bash

CFG="/home/rayc/Projects/shaper/configs/modelnet/comparenet/pointnet/pointnet_target_cls_last_fixseed.yaml"

for i in 0 1 2 3 4 5 6 7 8 9
do
    # for each support data, train 10 times and do ensemble
    for j in 0 1 2 3 4 5 6 7 8 9
    do
     CUDA_VISIBLE_DEVICES=1 python shaper_compare/tools/compare_train_net.py --cfg=$CFG\
     DATASET.COMPARE.CROSS_NUM ${i} OUTPUT_DIR "@_B/cross_${i}/REP_${j}"
    done
done
