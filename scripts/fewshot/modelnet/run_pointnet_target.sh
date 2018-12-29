#!/usr/bin/env bash
CFG="configs/fewshot/modelnet/pointnet_target_cls.yaml"
OUTPUT_DIR="outputs/fewshot/modelnet/pointnet_target_cls"

# cross validation
for i in {0..9}
do
   for j in {0..9}
   do
       echo "Cross $i / Replica $j"
       CUDA_VISIBLE_DEVICES=0 python shaper_fewshot/tools/train_net.py --cfg=$CFG  OUTPUT_DIR "@/cross_$i/rep_$j" DATASET.FEWSHOT.CROSS_INDEX $i
       CUDA_VISIBLE_DEVICES=0 python shaper_fewshot/tools/test_net.py --cfg=$CFG \
        --save-eval --save-pred OUTPUT_DIR "@/cross_$i/rep_$j" DATASET.FEWSHOT.CROSS_INDEX $i TEST.VOTE.ENABLE False
       CUDA_VISIBLE_DEVICES=0 python shaper_fewshot/tools/test_net.py --cfg=$CFG \
        --save-eval --suffix "multiview" OUTPUT_DIR "@/cross_$i/rep_$j" DATASET.FEWSHOT.CROSS_INDEX $i
  done
done

python shaper_fewshot/tools/analyze_accuracy.py -d $OUTPUT_DIR -f eval.pkl
python shaper_fewshot/tools/analyze_accuracy.py -d $OUTPUT_DIR -f eval_multiview.pkl
python shaper_fewshot/tools/ensemble_test.py --cfg=$CFG -d $OUTPUT_DIR -f pred.pkl