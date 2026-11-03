#!/bin/sh
# v37: v21 + tile-aligned wide view, trained 60 min on burst9-motion twins (cb9O exact per-tile translation = stepped / cb9H with the homography = clean; cb9A, the app aligner without H, fails grossly at 1024 px and is left out) + the v32/v33 twins and perfect sets; consistent labels, loose on texture.
# aligned by the app's homography (c32rot/c32roll P/C pairs, the burst11 case v29-v31 never saw), the c28 photo /
# text / chart pairs, perfect 3 MP photos and aliased text. Tile-aligned wide view (from v31), consistent slip
# labels, hard negatives next to slips. GEO_TAU_TEX (env) relaxes the slip rule on low-coherence texture.
# Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
until grep -q CACHE_OK data/rpb9_gen.log 2>/dev/null; do sleep 15; done
mkdir -p runs/v37
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_EDGE=0.15 RNN_GT_GEO_ALL=1 RNN_GT_GEO_TAU_TEX=${GEO_TAU_TEX:-1.5} RNN_TILE_CROP=1 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 RNN_W_NEAR=6 RNN_NEAR_R=8 RNN_W_PERFECT=${W_PERFECT:-3} \
EXTRA_TRAIN="cb9O_train:6,cb9H_train:6,c33O_train:2,c33A_train:2,c33H_train:2,c32rotP_train:3,c32rotC_train:4,c32rollP_train:3,c32rollC_train:3,c28p_train:3,c28t_train:2,c28c_train:2,c26g_train,c26z_train,c26p_train:3,cchartP0_train,cchartP1_train,cchartP2_train,cchartP3_train,c24p_train,c22txp_train:3,c20p_train" \
EXTRA_VAL="cb9O_val,cb9A_val,cb9H_val,c33O_val,c33A_val,c33H_val,c32rotP_val,c32rotC_val,c32rollP_val,c32rollC_val,c28p_val,c28t_val,c28c_val,c26g_val,c26z_val,c26p_val,cchart0_val,c22txp_val,c21txp_val,c20p_val" \
python train_v17.py --init ${INIT:-runs/v37/init.pt} --base runs/v22c/model.pt --tag v37 --out runs/v37 --steps 20000 \
    --minutes ${MINUTES:-60} --skip_s0 --eval_every 1000000 --lr 3e-4 --workers 2 > runs/v37/train.log 2>&1
echo TRAIN_OK
