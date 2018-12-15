#!/usr/bin/env bash

CFG="/home/rayc/Projects/shaper/configs/few_shot/pointnet/pointnet_fewshot_target_cls_baseline_pretrain_best.yaml"

# randomness
for i in 0 1 2 3 4 5 6 7 8 9
do
#   CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --cfg=$CFG  OUTPUT_DIR "@_$i"
    CUDA_VISIBLE_DEVICES=2 python tools/fewshot_train_net.py --cfg=$CFG\
    OUTPUT_DIR "@/REP_${i}"
#    echo $i
done
