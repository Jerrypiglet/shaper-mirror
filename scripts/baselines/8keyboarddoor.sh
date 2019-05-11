for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_keyboard.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_keyboard.yaml
    mv outputs/baselines/dgcnn_foveal_keyboard outputs/baselines/dgcnn_foveal_keyboard_$i
    mv outputs/baselines/dgcnn_instance_keyboard outputs/baselines/dgcnn_instance_keyboard_$i
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_door.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_door.yaml
    mv outputs/baselines/dgcnn_foveal_door outputs/baselines/dgcnn_foveal_door_$i
    mv outputs/baselines/dgcnn_instance_door outputs/baselines/dgcnn_instance_door_$i
done
