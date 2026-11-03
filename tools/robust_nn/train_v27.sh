#!/bin/sh
# v27 training only (data from gen_v27.sh): slipped outline pixels pushed to rejection (RNN_EDGE_HARD) AND
# well-aligned outline pixels protected as strongly as off-edge ones (RNN_EDGE_KEEP = RNN_W_NONEDGE), so only the
# misaligned edges go. Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
mkdir -p runs/v27
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_SETS=c24,c26,cchart RNN_GT_GEO_TAU=0.15 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 RNN_EDGE_HARD=2 RNN_EDGE_KEEP=3 \
EXTRA_TRAIN="cchartT0_train:3,cchartT1_train:2,cchartT2_train:2,cchartT3_train,cchartP0_train:2,cchartP1_train,cchartP2_train,cchartP3_train,c26g_train:3,c26p_train:2,c26z_train,c24g_train:2,c24p_train:2,c22txp_train:2,c20p_train" \
EXTRA_VAL="cchart0_val,cchart1_val,cchart2_val,cchart3_val,c26g_val,c26z_val,c26p_val,c24g_val,c22txp_val,c21txp_val,c20p_val" \
python train_v17.py --init runs/v22c/model.pt --base runs/v22c/model.pt --tag v27 --out runs/v27 --steps 6000 \
    --lr 3e-4 --eval_every 3000 --workers 2 > runs/v27/train.log 2>&1
echo TRAIN_OK
