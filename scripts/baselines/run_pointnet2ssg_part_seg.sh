export CUDA_VISIBLE_DEVICES=$1
python tools/train_part_seg.py --cfg=configs/baselines/pointnet2ssg_part_seg.yaml
