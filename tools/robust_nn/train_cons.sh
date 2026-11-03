#!/bin/sh
# v25 prototype (~20 min): v22 + 4 burst-consistency inputs (consensus_features), below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
mkdir -p runs/v25
CONSF=1 RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_SETS=c24 RNN_GT_GEO_TAU=0.15 RNN_W_TEX=1 RNN_W_NONEDGE=3 \
RNN_MISS_VISIBLE=1 EXTRA_TRAIN="c24g_train:4,c24z_train:4,c24p_train:3,c22txp_train,c22scr_train,c20p_train" \
EXTRA_VAL="c24p_val,c24g_val,c24z_val" \
python train_v17.py --init runs/v22c/model.pt --widen --base runs/v22c/model.pt --tag v25 --out runs/v25 --steps 3500 \
    --lr 3e-4 --eval_every 3500 > runs/v25/train.log 2>&1
echo "EXIT $?" >> runs/v25/train.log
