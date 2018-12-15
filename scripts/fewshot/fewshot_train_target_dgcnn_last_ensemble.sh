#!/usr/bin/env bash

CFG="/home/rayc/Projects/shaper/configs/few_shot/dgcnn_fewshot_target_cls_last.yaml"

for i in 0 1 2 3 4 5 6 7 8 9
do
    # for each support data, train 5 times and do ensemble
    for j in 0 1 2 3 4
    do
     CUDA_VISIBLE_DEVICES=0 python tools/fewshot_train_net.py --cfg=$CFG\
     DATASET.FEW_SHOT.CROSS_NUM ${i} OUTPUT_DIR "@/cross_${i}/REP_${j}" TRAIN.FREEZE False
    done
done
