import os, sys


with open('sorted_cats.txt') as f:
    cats = [line.lower().strip() for line in f.readlines()]

cat = cats[int(sys.argv[1])]
print(cat)


out_fn = 'outputs/baselines/dgcnn_instance_%s_large'%cat
config_fn = out_fn.replace('outputs','configs')
for i in range(5):
    if os.path.exists(out_fn+'_%d'%i):
        print(i)
        continue
    else:
        a=os.system('python tools/train_ins_seg.py --cfg=%s.yaml'% config_fn)
        if a == 0:
            os.rename(out_fn, out_fn + '_%d'%i)

out_fn = 'outputs/baselines/dgcnn_instance_%s'%cat
config_fn = out_fn.replace('outputs','configs')
for i in range(5):
    if os.path.exists(out_fn+'_%d'%i):
        print(i)
        continue
    else:
        a=os.system('python tools/train_ins_seg.py --cfg=%s.yaml'% config_fn)
        if a == 0:
            os.rename(out_fn, out_fn + '_%d'%i)


