#!/usr/bin/env bash
CFG="configs/baselines/pointnet2ssg_part_seg.yaml"
FLAG="-t part_segmentation"

for i in 1
do
   CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --cfg=$CFG $FLAG OUTPUT_DIR "@_$i"
   CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg=$CFG $FLAG OUTPUT_DIR "@_$i" \
    INPUT.NUM_POINTS 3000 \
    DATALOADER.NUM_WORKERS 4 \
    DATASET.ROOT_DIR "data/shapenet" \
    DATASET.TYPE "ShapeNet"
done
