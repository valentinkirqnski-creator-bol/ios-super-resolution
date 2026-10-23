#!/bin/sh
# v23 data: the "same edge slightly rotated" artifact - 24-64 px segments on the strongest straight outlines turn
# 0.5-3 deg (rotation, or rotation + tiny shift) the same way in 40-95% of the frames; perfect alignment otherwise
# (tp_real oracle), real photos (er) and aliased text (ert). Then v23 is fine-tuned from v22 with the geometry rule on
# outlines (RNN_GT_GEO_EDGE: the edge's own displacement across itself, tau 0.15 raw px).
#   sh gen_v23.sh > data/rp23_gen.log 2>&1
set -e
cd "$(dirname "$0")"
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 6"
ER="RNN_ROT_SMALL=6,14 RNN_EDGE_ROT=1"
env RNN_PERFECT=still RNN_ROT_PATCHES=0,0,0,0 $ER $G --out data/rp23er_train --n 200 --seed0 2700000
env RNN_PERFECT=still RNN_ROT_PATCHES=0,0,0,0 $ER $G --out data/rp23er_val --n 30 --seed0 9940000 --val
env RNN_TEXT=1 RNN_TEXT_CAP=1.8,9 $ER $G --out data/rp23ert_train --n 120 --seed0 2800000
env RNN_TEXT=1 RNN_TEXT_CAP=1.8,9 $ER $G --out data/rp23ert_val --n 20 --seed0 9950000 --val
echo GEN_OK
C="python precompute_cache.py --workers 5 --oracle"
$C --src data/rp23er_val --out data/c23er_val --val --no_corrupt
$C --src data/rp23ert_val --out data/c23ert_val --val --no_corrupt
$C --src data/rp23er_train --out data/c23er_train --variants 2 --no_corrupt
$C --src data/rp23ert_train --out data/c23ert_train --variants 2 --no_corrupt
echo CACHE_OK
mkdir -p runs/v23
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_EDGE=0.15 RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 \
EXTRA_TRAIN="c23er_train:2,c23ert_train:2,c22txp_train,c22txm_train,c22txr_train,c22scr_train,c21txp_train,c20pH_train,c20p_train" \
EXTRA_VAL="c23er_val,c23ert_val,c22txp_val,c22scr_val,c20pH_val,c20p_val" \
python train_v17.py --init runs/v22c/model.pt --base runs/v22c/model.pt --tag v23 --out runs/v23 --steps 8000 --lr 2e-4 --eval_every 2000 > runs/v23/train.log 2>&1
echo TRAIN_OK
