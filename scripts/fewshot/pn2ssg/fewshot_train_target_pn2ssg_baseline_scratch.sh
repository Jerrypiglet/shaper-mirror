#!/usr/bin/env bash

CFG="/home/rayc/Projects/shaper/configs/few_shot/pn2ssg/pointnet2ssg_fewshot_target_cls_ap_baseline_scratch.yaml"

for i in 0 1 2 3 4 5 6 7 8 9
do
   CUDA_VISIBLE_DEVICES=1 python tools/fewshot_train_net.py --cfg=$CFG\
   OUTPUT_DIR "@/REP_${i}"
done
