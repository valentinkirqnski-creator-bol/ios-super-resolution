#!/bin/sh
# v38: v21 (plain, shipping kernel) trained 60 min on full-12 MP tiled twins (c38S perfect / c38M burst9 motion / c38C motion + patches + tile slips) + the earlier twins and perfect sets.
# aligned by the app's homography (c32rot/c32roll P/C pairs, the burst11 case v29-v31 never saw), the c28 photo /
# text / chart pairs, perfect 3 MP photos and aliased text. Tile-aligned wide view (from v31), consistent slip
# labels, hard negatives next to slips. GEO_TAU_TEX (env) relaxes the slip rule on low-coherence texture.
# Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
until grep -q CACHE_OK data/rp38_gen.log 2>/dev/null; do sleep 15; done
mkdir -p runs/v38
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_EDGE=0.15 RNN_GT_GEO_ALL=1 RNN_GT_GEO_TAU_TEX=${GEO_TAU_TEX:-1.5} RNN_TILE_CROP=1 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 RNN_W_NEAR=6 RNN_NEAR_R=8 RNN_W_PERFECT=${W_PERFECT:-3} \
EXTRA_TRAIN="c38S_train:3,c38M_train:4,c38C_train:6,cb9O_train:2,cb9H_train:2,c33O_train:2,c33A_train:2,c33H_train:2,c32rotP_train:3,c32rotC_train:4,c32rollP_train:3,c32rollC_train:3,c28p_train:3,c28t_train:2,c28c_train:2,c26g_train,c26z_train,c26p_train:3,cchartP0_train,cchartP1_train,cchartP2_train,cchartP3_train,c24p_train,c22txp_train:3,c20p_train" \
EXTRA_VAL="c38S_val,c38M_val,c38C_val,cb9O_val,cb9H_val,c33O_val,c33A_val,c33H_val,c32rotP_val,c32rotC_val,c32rollP_val,c32rollC_val,c28p_val,c28t_val,c28c_val,c26g_val,c26z_val,c26p_val,cchart0_val,c22txp_val,c21txp_val,c20p_val" \
python train_v17.py --init ${INIT:-runs/v21/model.pt} --base runs/v22c/model.pt --tag v38 --out runs/v38 --steps 20000 \
    --minutes ${MINUTES:-60} --skip_s0 --eval_every 1000000 --lr 3e-4 --workers 2 > runs/v38/train.log 2>&1
echo TRAIN_OK
