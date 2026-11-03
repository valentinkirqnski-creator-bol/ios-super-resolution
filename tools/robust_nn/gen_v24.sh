#!/bin/sh
# v24: 768 px, 8-frame bursts at camera pixel scale (real photos 1:1, distant aliased text), perfectly aligned with
# the true motion per tile (tp_real oracle). Three kinds of training sample:
#   p : perfectly aligned - keep everything (super-resolution + denoising)
#   g : ghosted / thicker edges - alignment vectors of tiles on edges nudged 0.2-1.0 px + straight-edge steps
#   z : zippery edges - a 0.5-2.5 deg camera roll under one translation per tile: each tile's piece of a straight
#       edge turns slightly and the edge steps at tile borders
# Then v24 is fine-tuned from v22 with the geometry rule on outlines for these sets.
#   sh gen_v24.sh > data/rp24_gen.log 2>&1
set -e
cd "$(dirname "$0")"
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_SIZE=768 RNN_NFRAMES=8 TP_SMALL_RANGE=0.2,1.0
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 6"
PH="RNN_PERFECT=still RNN_ROT_PATCHES=0,0,0,0 RNN_TEX_SCALE=0.5"
TX="RNN_TEXT=1 RNN_TEXT_CAP=1.8,6"
env $PH $G --out data/rp24ph_train --n 90 --seed0 3000000
env $TX $G --out data/rp24tx_train --n 60 --seed0 3100000
env RNN_ROT_PATCHES=0,0,0,0 RNN_TEX_SCALE=0.5 RNN_ROLL=0.5,2.5 $G --out data/rp24z_train --n 70 --seed0 3200000
env $TX RNN_ROT_PATCHES=0,0,0,0 RNN_ROLL=0.5,2.5 $G --out data/rp24z_train --n 30 --seed0 3300000
env $PH $G --out data/rp24ph_val --n 8 --seed0 9600000 --val
env $TX $G --out data/rp24tx_val --n 6 --seed0 9610000 --val
env RNN_ROT_PATCHES=0,0,0,0 RNN_TEX_SCALE=0.5 RNN_ROLL=0.5,2.5 $G --out data/rp24z_val --n 5 --seed0 9620000 --val
env $TX RNN_ROT_PATCHES=0,0,0,0 RNN_ROLL=0.5,2.5 $G --out data/rp24z_val --n 3 --seed0 9630000 --val
echo GEN_OK
C="python precompute_cache.py --workers 5 --oracle"
$C --src data/rp24ph_val --out data/c24p_val --val --no_corrupt
$C --src data/rp24tx_val --out data/c24pt_val --val --no_corrupt
$C --src data/rp24ph_val --out data/c24g_val --val --val_corrupt_only --small --edge_step
$C --src data/rp24tx_val --out data/c24gt_val --val --val_corrupt_only --small --edge_step
$C --src data/rp24z_val --out data/c24z_val --val --no_corrupt
$C --src data/rp24ph_train,data/rp24tx_train --out data/c24p_train --variants 1 --no_corrupt
$C --src data/rp24ph_train,data/rp24tx_train --out data/c24g_train --variants 2 --corrupt_p 1.0 --small --edge_step
$C --src data/rp24z_train --out data/c24z_train --variants 2 --no_corrupt
echo CACHE_OK
mkdir -p runs/v24
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_SETS=c24 RNN_GT_GEO_TAU=0.15 RNN_W_TEX=1 RNN_W_NONEDGE=3 \
RNN_MISS_VISIBLE=1 \
EXTRA_TRAIN="c24p_train:2,c24g_train:2,c24z_train:3,c22txp_train,c22scr_train,c21txp_train,c20pH_train,c20p_train" \
EXTRA_VAL="c24p_val,c24pt_val,c24g_val,c24gt_val,c24z_val,c22txp_val,c20pH_val,c20p_val" \
python train_v17.py --init runs/v22c/model.pt --base runs/v22c/model.pt --tag v24 --out runs/v24 --steps 10000 \
    --lr 2e-4 --eval_every 2500 > runs/v24/train.log 2>&1
echo TRAIN_OK
