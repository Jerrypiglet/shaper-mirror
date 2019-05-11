for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_vase.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_vase.yaml
    mv outputs/baselines/dgcnn_foveal_vase outputs/baselines/dgcnn_foveal_vase_$i
    mv outputs/baselines/dgcnn_instance_vase outputs/baselines/dgcnn_instance_vase_$i
done
