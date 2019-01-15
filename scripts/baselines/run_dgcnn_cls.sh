#!/usr/bin/env bash
CFG="configs/baselines/dgcnn_cls.yaml"

# randomness
for i in 1 2 3
do
   CUDA_VISIBLE_DEVICES=0 python tools/train_cls.py --cfg=$CFG OUTPUT_DIR "@_$i"
   CUDA_VISIBLE_DEVICES=0 python tools/test_cls.py --cfg=$CFG OUTPUT_DIR "@_$i"
done
