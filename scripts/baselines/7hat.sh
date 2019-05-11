for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_hat.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_hat.yaml
    mv outputs/baselines/dgcnn_foveal_hat outputs/baselines/dgcnn_foveal_hat_$i
    mv outputs/baselines/dgcnn_instance_hat outputs/baselines/dgcnn_instance_hat_$i
done
