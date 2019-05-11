for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_microwave.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_microwave.yaml
    mv outputs/baselines/dgcnn_foveal_microwave outputs/baselines/dgcnn_foveal_microwave_$i
    mv outputs/baselines/dgcnn_instance_microwave outputs/baselines/dgcnn_instance_microwave_$i
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_bag.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_bag.yaml
    mv outputs/baselines/dgcnn_foveal_bag outputs/baselines/dgcnn_foveal_bag_$i
    mv outputs/baselines/dgcnn_instance_bag outputs/baselines/dgcnn_instance_bag_$i
done
