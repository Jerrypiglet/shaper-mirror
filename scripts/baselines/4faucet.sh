for i in `seq 0 4`
do
    python tools/train_foveal_seg.py --cfg=configs/baselines/dgcnn_foveal_faucet.yaml
    python tools/train_ins_seg.py --cfg=configs/baselines/dgcnn_instance_faucet.yaml
    mv outputs/baselines/dgcnn_foveal_faucet outputs/baselines/dgcnn_foveal_faucet_$i
    mv outputs/baselines/dgcnn_instance_faucet outputs/baselines/dgcnn_instance_faucet_$i
done
