#!/bin/sh
# v28 training: waits for the rebuilt caches (cache_v28.sh), then ~27 min of training + the final report vs v22c
# (~40 min total). Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
until grep -q CACHE_OK data/rp28_cache.log 2>/dev/null; do sleep 15; done
mkdir -p runs/v28
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_SETS=c24,c26,cchart,c28 RNN_GT_GEO_TAU=0.15 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 \
EXTRA_TRAIN="c28p_train:4,c28t_train:3,c28c_train:3,c26g_train,c26z_train,c26p_train:2,cchartP0_train,cchartP1_train,cchartP2_train,cchartP3_train,c24p_train,c22txp_train:2,c20p_train" \
EXTRA_VAL="c28p_val,c28t_val,c28c_val,c26g_val,c26z_val,c26p_val,cchart0_val,cchart2_val,c22txp_val,c21txp_val,c20p_val" \
python train_v17.py --init runs/v22c/model.pt --ctx 8 --base runs/v22c/model.pt --tag v28 --out runs/v28 --steps 20000 \
    --minutes 27 --skip_s0 --eval_every 1000000 --lr 5e-4 --workers 2 > runs/v28/train.log 2>&1
echo TRAIN_OK
