import os, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp',type=int)
parser.add_argument('--cat', type=int)
parser.add_argument('--iter', type=int)
FLAGS=parser.parse_args()

with open('sorted_cats.txt') as f:
    cats = [line.lower().strip() for line in f.readlines()]

cat = cats[FLAGS.cat]
print(cat)

if FLAGS.exp == 0:
    out_fn = 'outputs/baselines/dgcnn_instance_%s'%cat
elif FLAGS.exp==1:
    out_fn = 'outputs/baselines/dgcnn_instance_%s_large'%cat
elif FLAGS.exp == 2:
    out_fn = 'outputs/baselines/dgcnn_foveal_%s'%cat
else:
    print('invalid exp')
    exit(0)

config_fn = out_fn.replace('outputs','configs')
out_fn+='_%d'%FLAGS.iter

if os.path.exists(out_fn):
    print('Already exists')
else:
    a=os.system('rm -rf %s && python tools/train_ins_seg.py --cfg=%s.yaml OUTPUT_DIR %s'% (out_fn, config_fn, out_fn))
