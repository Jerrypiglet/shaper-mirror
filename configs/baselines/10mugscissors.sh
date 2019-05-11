for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_mug.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_mug.yaml
    mv outputs/baselines/dgcnn_foveal_mug outputs/baselines/dgcnn_foveal_mug_$i
    mv outputs/baselines/dgcnn_instance_mug outputs/baselines/dgcnn_instance_mug_$i
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_scissors.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_scissors.yaml
    mv outputs/baselines/dgcnn_foveal_scissors outputs/baselines/dgcnn_foveal_scissors_$i
    mv outputs/baselines/dgcnn_instance_scissors outputs/baselines/dgcnn_instance_scissors_$i
done
