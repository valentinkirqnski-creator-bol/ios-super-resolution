#!/bin/sh
# v31: v30 + hard negatives (RNN_W_NEAR: good pixels next to a slip cost more to drop), consistent slip labels (RNN_GT_GEO_ALL), crops on the
# tile grid (RNN_TILE_CROP). Same sets as v29. Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
mkdir -p runs/v31
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_EDGE=0.15 RNN_GT_GEO_ALL=1 RNN_TILE_CROP=1 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 RNN_W_NEAR=6 RNN_NEAR_R=8 \
EXTRA_TRAIN="c28p_train:4,c28t_train:3,c28c_train:3,c26g_train,c26z_train,c26p_train:2,cchartP0_train,cchartP1_train,cchartP2_train,cchartP3_train,c24p_train,c22txp_train:2,c20p_train" \
EXTRA_VAL="c28p_val,c28t_val,c28c_val,c26g_val,c26z_val,c26p_val,cchart0_val,cchart2_val,c22txp_val,c21txp_val,c20p_val" \
python train_v17.py --init runs/v30/model_s5132.pt --base runs/v22c/model.pt --tag v31 --out runs/v31 --steps 20000 \
    --minutes ${MINUTES:-30} --skip_s0 --eval_every 1000000 --lr 3e-4 --workers 2 > runs/v31/train.log 2>&1
echo TRAIN_OK
