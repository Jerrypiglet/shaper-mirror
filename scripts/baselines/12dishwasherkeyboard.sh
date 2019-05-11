for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_keyboard.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_keyboard.yaml
    mv outputs/baselines/dgcnn_foveal_keyboard outputs/baselines/dgcnn_foveal_keyboard_$i
    mv outputs/baselines/dgcnn_instance_keyboard outputs/baselines/dgcnn_instance_keyboard_$i
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_dishwasher.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_dishwasher.yaml
    mv outputs/baselines/dgcnn_foveal_dishwasher outputs/baselines/dgcnn_foveal_dishwasher_$i
    mv outputs/baselines/dgcnn_instance_dishwasher outputs/baselines/dgcnn_instance_dishwasher_$i
done
