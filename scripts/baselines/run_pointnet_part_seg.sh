#!/usr/bin/env bash
CFG="configs/baselines/pointnet_part_seg.yaml"

for i in 1 2 3
do
   CUDA_VISIBLE_DEVICES=0,1 python tools/train_part_seg.py --cfg=$CFG OUTPUT_DIR "@_$i"
   CUDA_VISIBLE_DEVICES=0 python tools/test_part_seg.py --cfg=$CFG OUTPUT_DIR "@_$i" \
    INPUT.NUM_POINTS 3000 \
    DATALOADER.NUM_WORKERS 4 \
    DATASET.ROOT_DIR "data/shapenet" \
    DATASET.TYPE "ShapeNet"
done
