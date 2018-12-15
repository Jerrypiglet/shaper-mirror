#!/usr/bin/env bash

CFG="/home/rayc/Projects/shaper/configs/few_shot/pointnet2msg_fewshot_target_cls_best.yaml"

# randomness
for i in 0 1 2 3 4 5 6 7 8 9
do
   CUDA_VISIBLE_DEVICES=1 python tools/fewshot_train_net.py --cfg=$CFG\
     DATASET.FEW_SHOT.CROSS_NUM $i OUTPUT_DIR "@/best_freeze_$i" TRAIN.FREEZE True
done
