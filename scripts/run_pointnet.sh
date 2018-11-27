#!/usr/bin/env bash
CFG="configs/baselines/pointnet_cls.yaml"

# randomness
for i in 1 2 3 4 5
do
   CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --cfg=$CFG OUTPUT_DIR "@_$i"
done
