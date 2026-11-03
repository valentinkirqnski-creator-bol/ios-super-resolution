#!/bin/sh
# v26: 3 MP (1728 x 1728) 8-frame bursts of real photos at camera scale with the two artifacts - ghosted / thicker
# edges (edge-tile vector nudges 0.2-1.0 px + straight-edge steps) and zippery edges (0.5-2.5 deg roll under one
# translation per tile) - then v22 + the 3 edge-line inputs (EDGEF) trained ~45 min. Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rp26_gen.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_SIZE=1728 RNN_NFRAMES=8 TP_SMALL_RANGE=0.2,1.0
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 3"
PH="RNN_PERFECT=still RNN_ROT_PATCHES=0,0,0,0 RNN_TEX_SCALE=0.5"
Z="RNN_ROT_PATCHES=0,0,0,0 RNN_TEX_SCALE=0.5 RNN_ROLL=0.5,2.5"
env $PH $G --out data/rp26ph_train --n 8 --seed0 3400000
env $Z $G --out data/rp26z_train --n 8 --seed0 3500000
env $PH $G --out data/rp26ph_val --n 2 --seed0 9640000 --val
env $Z $G --out data/rp26z_val --n 2 --seed0 9650000 --val
echo GEN_OK
C="python precompute_cache.py --workers 2 --oracle"
$C --src data/rp26ph_val --out data/c26p_val --val --no_corrupt
$C --src data/rp26ph_val --out data/c26g_val --val --val_corrupt_only --small --edge_step
$C --src data/rp26z_val --out data/c26z_val --val --no_corrupt
$C --src data/rp26ph_train --out data/c26g_train --variants 2 --corrupt_p 1.0 --small --edge_step
$C --src data/rp26ph_train --out data/c26p_train --variants 1 --no_corrupt
$C --src data/rp26z_train --out data/c26z_train --variants 1 --no_corrupt
echo CACHE_OK
mkdir -p runs/v26
EDGEF=1 EDGEF_L=8 EDGEF_BAND=1 RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_SETS=c24,c26 RNN_GT_GEO_TAU=0.15 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 \
EXTRA_TRAIN="c26g_train:4,c26z_train:4,c26p_train:3,c24g_train:2,c24z_train:2,c24p_train:2,c22txp_train,c20p_train" \
EXTRA_VAL="c26g_val,c26z_val,c26p_val,c24g_val,c24z_val" \
python train_v17.py --init runs/v22c/model.pt --widen --base runs/v22c/model.pt --tag v26 --out runs/v26 --steps 8000 \
    --lr 3e-4 --eval_every 4000 --workers 2 > runs/v26/train.log 2>&1
echo TRAIN_OK
