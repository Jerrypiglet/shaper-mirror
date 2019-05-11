for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_knife.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_knife.yaml
    mv outputs/baselines/dgcnn_foveal_knife outputs/baselines/dgcnn_foveal_knife_$i
    mv outputs/baselines/dgcnn_instance_knife outputs/baselines/dgcnn_instance_knife_$i
done
