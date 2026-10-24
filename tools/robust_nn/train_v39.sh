#!/bin/sh
# v39: content-aware mask - the pixel-seeing PixNet (the app's v15 kernels: pixels of the reference and the aligned
# frame through dilated convs, 17x17 view) over a frozen v21 base, trained on the 12 MP twins, the earlier twins and
# the perfect sets with the consistent slip labels. Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
mkdir -p runs/v39
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_EDGE=0.15 RNN_GT_GEO_ALL=1 RNN_GT_GEO_TAU_TEX=1.5 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 RNN_W_NEAR=6 RNN_NEAR_R=8 RNN_W_PERFECT=3 \
EXTRA_TRAIN="c38S_train:3,c38M_train:4,c38C_train:6,cb9O_train:2,cb9H_train:2,c33O_train:2,c33H_train:2,c32rotP_train:2,c32rotC_train:3,c32rollP_train:2,c32rollC_train:3,c28p_train:3,c28t_train:2,c28c_train:2,c26g_train,c26p_train:3,c22txp_train:3,c20p_train" \
EXTRA_VAL="c38S_val,c38M_val,c38C_val,cb9O_val,c28t_val,c26g_val,c26p_val,c22txp_val" \
python train_pixnet.py --base runs/v21/model.pt --tag v39 --out runs/v39 --minutes ${MINUTES:-40} --lr 1e-3 --workers 2 > runs/v39/train.log 2>&1
echo TRAIN_OK
