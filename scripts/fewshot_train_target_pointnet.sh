#!/usr/bin/env bash

CFG="configs/few_shot/pointnet_fewshot_target_cls.yaml"

# randomness
for i in 0 1 2 3 4 5 6 7 8 9
do
#   CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --cfg=$CFG  OUTPUT_DIR "@_$i"
   CUDA_VISIBLE_DEVICES=1 python tools/fewshot_train_net.py --cfg=$CFG\
     DATASET.FEW_SHOT.CROSS_NUM $i OUTPUT_DIR "@_freeze_$i"
#    echo $i
done
