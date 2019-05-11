for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_chair.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_chair.yaml
    mv outputs/baselines/dgcnn_foveal_chair outputs/baselines/dgcnn_foveal_chair_$i
    mv outputs/baselines/dgcnn_instance_chair outputs/baselines/dgcnn_instance_chair_$i
done
