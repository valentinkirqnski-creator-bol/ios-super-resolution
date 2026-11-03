#!/bin/sh
# v24 reduced to ~40 min at low load: 3 workers, below-normal priority (set by the caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rp24b_gen.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_SIZE=768 RNN_NFRAMES=8 TP_SMALL_RANGE=0.2,1.0
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 3"
Z="RNN_ROT_PATCHES=0,0,0,0 RNN_TEX_SCALE=0.5 RNN_ROLL=0.5,2.5"
env $Z $G --out data/rp24z_train --n 30 --seed0 3200000
env $Z $G --out data/rp24z_val --n 4 --seed0 9620000 --val
env RNN_PERFECT=still RNN_ROT_PATCHES=0,0,0,0 RNN_TEX_SCALE=0.5 $G --out data/rp24ph_val --n 4 --seed0 9600000 --val
echo GEN_OK
C="python precompute_cache.py --workers 3 --oracle"
$C --src data/rp24ph_val --out data/c24p_val --val --no_corrupt
$C --src data/rp24ph_val --out data/c24g_val --val --val_corrupt_only --small --edge_step
$C --src data/rp24z_val --out data/c24z_val --val --no_corrupt
$C --src data/rp24ph_train --limit 30 --out data/c24p_train --variants 1 --no_corrupt
$C --src data/rp24tx_train --limit 10 --out data/c24p_train --variants 1 --no_corrupt
$C --src data/rp24ph_train --limit 30 --out data/c24g_train --variants 1 --corrupt_p 1.0 --small --edge_step
$C --src data/rp24tx_train --limit 10 --out data/c24g_train --variants 1 --corrupt_p 1.0 --small --edge_step
$C --src data/rp24z_train --out data/c24z_train --variants 1 --no_corrupt
echo CACHE_OK
mkdir -p runs/v24
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_SETS=c24 RNN_GT_GEO_TAU=0.15 RNN_W_TEX=1 RNN_W_NONEDGE=3 \
RNN_MISS_VISIBLE=1 EXTRA_TRAIN="c24p_train:3,c24g_train:3,c24z_train:4,c22txp_train,c22scr_train,c20p_train" \
EXTRA_VAL="c24p_val,c24g_val,c24z_val" \
python train_v17.py --init runs/v22c/model.pt --base runs/v22c/model.pt --tag v24 --out runs/v24 --steps 3000 \
    --lr 2e-4 --eval_every 3000 > runs/v24/train.log 2>&1
echo TRAIN_OK
