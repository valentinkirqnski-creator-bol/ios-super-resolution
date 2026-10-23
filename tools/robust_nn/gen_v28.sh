#!/bin/sh
# v28: many deliberately misaligned 8-frame bursts next to their perfectly aligned version (the target compares the
# noise-free merge with the used vs the true alignment): tile vectors nudged 0.2-2.0 px (uniform, ANY area) plus
# content patches (24-96 px, half on straight edges, half anywhere) translated / rotated / both by 0.2-2.0 px of
# displacement, randomly per frame or the same way in most frames (ghost / double / zippery edges / blur), on real
# photos (camera scale), aliased small text and edge charts. Model: v22c + the wide-view context branch (--ctx 8).
# Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rp28_gen.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_NFRAMES=8 RNN_ROT_PATCHES=0,0,0,0 RNN_ROT_SMALL=6,14 RNN_SMALL_DISP=0.2,2.0,0.5,0.7
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 2"
env RNN_SIZE=1024 RNN_PERFECT=still RNN_TEX_SCALE=0.5 $G --out data/rp28p_train --n 12 --seed0 3700000
env RNN_SIZE=1024 RNN_PERFECT=still RNN_TEX_SCALE=0.5 $G --out data/rp28p_val --n 2 --seed0 9810000 --val
env RNN_SIZE=768 RNN_TEXT=1 RNN_TEXT_CAP=1.8,6 $G --out data/rp28t_train --n 8 --seed0 3710000
env RNN_SIZE=768 RNN_TEXT=1 RNN_TEXT_CAP=1.8,6 $G --out data/rp28t_val --n 2 --seed0 9820000 --val
i=0
for c in "1,2" "3,6" "10,20" "40,120"; do
  env RNN_SIZE=768 RNN_TEX_SCALE=4 RNN_CHART=$c $G --out data/rp28c_train --n 2 --seed0 $((3720000 + i * 10))
  i=$((i + 1))
done
env RNN_SIZE=768 RNN_TEX_SCALE=4 RNN_CHART=1,3 $G --out data/rp28c_val --n 1 --seed0 9830000 --val
env RNN_SIZE=768 RNN_TEX_SCALE=4 RNN_CHART=8,30 $G --out data/rp28c_val --n 1 --seed0 9830010 --val
echo GEN_OK
export TP_SMALL_RANGE=0.2,2.0 TP_SMALL_ANY=1 TP_SMALL_UNIFORM=1 TP_SMALL_CONSISTENT=1
C="python precompute_cache.py --workers 2 --oracle"
for k in p t c; do
  $C --src data/rp28${k}_val --out data/c28${k}_val --val --val_corrupt_only --small
  $C --src data/rp28${k}_train --out data/c28${k}_train --variants 2 --corrupt_p 0.8 --small
done
echo CACHE_OK
mkdir -p runs/v28
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_SETS=c24,c26,cchart,c28 RNN_GT_GEO_TAU=0.15 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 \
EXTRA_TRAIN="c28p_train:4,c28t_train:3,c28c_train:3,c26g_train,c26z_train,c26p_train:2,cchartP0_train,cchartP1_train,cchartP2_train,cchartP3_train,c24p_train,c22txp_train:2,c20p_train" \
EXTRA_VAL="c28p_val,c28t_val,c28c_val,c26g_val,c26z_val,c26p_val,cchart0_val,cchart2_val,c22txp_val,c21txp_val,c20p_val" \
python train_v17.py --init runs/v22c/model.pt --ctx 8 --base runs/v22c/model.pt --tag v28 --out runs/v28 --steps 8000 \
    --lr 5e-4 --eval_every 4000 --workers 2 > runs/v28/train.log 2>&1
echo TRAIN_OK
