for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_bottle.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_bottle.yaml
    mv outputs/baselines/dgcnn_foveal_bottle outputs/baselines/dgcnn_foveal_bottle_$i
    mv outputs/baselines/dgcnn_instance_bottle outputs/baselines/dgcnn_instance_bottle_$i
done
