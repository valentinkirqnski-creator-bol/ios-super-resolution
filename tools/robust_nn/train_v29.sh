#!/bin/sh
# v29: v28 (wide view) trained 30 more minutes with ONE consistent slip label in every set and on all structure
# (RNN_GT_GEO_ALL=1, tau 0.15 raw px; no per-set RNN_GT_GEO_SETS any more). Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
mkdir -p runs/v29
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_EDGE=0.15 RNN_GT_GEO_ALL=1 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 \
EXTRA_TRAIN="c28p_train:4,c28t_train:3,c28c_train:3,c26g_train,c26z_train,c26p_train:2,cchartP0_train,cchartP1_train,cchartP2_train,cchartP3_train,c24p_train,c22txp_train:2,c20p_train" \
EXTRA_VAL="c28p_val,c28t_val,c28c_val,c26g_val,c26z_val,c26p_val,cchart0_val,cchart2_val,c22txp_val,c21txp_val,c20p_val" \
python train_v17.py --init runs/v28/model_s6096.pt --base runs/v22c/model.pt --tag v29 --out runs/v29 --steps 20000 \
    --minutes 30 --skip_s0 --eval_every 1000000 --lr 3e-4 --workers 2 > runs/v29/train.log 2>&1
echo TRAIN_OK
