export CUDA_VISIBLE_DEVICES=$1
python tools/train_part_seg.py --cfg=configs/baselines/dgcnn_part_seg.yaml
