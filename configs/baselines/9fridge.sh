for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_refrigerator.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_refrigerator.yaml
    mv outputs/baselines/dgcnn_foveal_refrigerator outputs/baselines/dgcnn_foveal_refrigerator_$i
    mv outputs/baselines/dgcnn_instance_refrigerator outputs/baselines/dgcnn_instance_refrigerator_$i
done
