for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_lamp.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_lamp.yaml
    mv outputs/baselines/dgcnn_foveal_lamp outputs/baselines/dgcnn_foveal_lamp_$i
    mv outputs/baselines/dgcnn_instance_lamp outputs/baselines/dgcnn_instance_lamp_$i
done
