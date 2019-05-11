for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_storage_furniture.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_storage_furniture.yaml
    mv outputs/baselines/dgcnn_foveal_storage_furniture outputs/baselines/dgcnn_foveal_storage_furniture_$i
    mv outputs/baselines/dgcnn_instance_storage_furniture outputs/baselines/dgcnn_instance_storage_furniture_$i
done
