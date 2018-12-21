#!/usr/bin/env bash

CFG="/home/rayc/Projects/shaper/configs/few_shot/pointnet/pointnet_fewshot_target_cls_onelayer_freeze_last.yaml"

for i in 0 1 2 3 4 5 6 7 8 9
do
    # for each support data, train 10 times and do ensemble
    for j in 0 1 2 3 4 5 6 7 8 9
    do
     CUDA_VISIBLE_DEVICES=1 python tools/fewshot_train_net.py --cfg=$CFG\
     DATASET.MODELNET_FEWSHOT.CROSS_NUM ${i} OUTPUT_DIR "@/cross_${i}/REP_${j}"
    done
done
