#!/usr/bin/env bash
CFG="configs/fewshot/modelnet/pointnet_cls.yaml"
OUTPUT_DIR="outputs/fewshot/modelnet/pointnet_cls"

# cross validation
for i in {0..9}
do
   echo "Cross $i"
   CUDA_VISIBLE_DEVICES=0 python shaper_fewshot/tools/train_net.py --cfg=$CFG  OUTPUT_DIR "@_$i"
   CUDA_VISIBLE_DEVICES=0 python shaper_fewshot/tools/test_net.py --cfg=$CFG  \
    --save-eval --suffix "best_multiview" OUTPUT_DIR "@_$i"  TEST.WEIGHT "@/model_best.pth"
done

python shaper_fewshot/tools/analyze_accuracy.py -d $OUTPUT_DIR -f eval_multiview.pkl