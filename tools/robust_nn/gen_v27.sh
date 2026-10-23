#!/bin/sh
# v27: v22c fine-tuned so SMALL edge slips (0.2-1 px, the thicker / ghosted edge) are rejected outright instead of
# half-weighted: edge-chart bursts (thin lines 1-2 px .. blocks 40-120 px) with slipped edge tiles + their perfect
# twins, the 3 MP ghosted / perfect sets, aliased text and the old sets; loss RNN_EDGE_HARD (slipped outline pixels
# -> target 0, extra weight). Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rp27_gen.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
i=0
for c in "1,2" "3,6" "10,20" "40,120"; do
  RNN_SIZE=1024 RNN_NFRAMES=8 RNN_TEX_SCALE=4 RNN_CHART=$c python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 \
      --workers 2 --out data/rpchartT$i --n 3 --seed0 $((3600000 + i * 100))
  TP_SMALL_RANGE=0.2,1.0 python precompute_cache.py --workers 2 --oracle --src data/rpchartT$i --out data/cchartT${i}_train \
      --variants 2 --corrupt_p 1.0 --small
  python precompute_cache.py --workers 2 --oracle --src data/rpchartT$i --out data/cchartP${i}_train --variants 1 --no_corrupt
  i=$((i + 1))
done
echo CACHE_OK
mkdir -p runs/v27
RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.2,0.7 RNN_GT_GEO_SETS=c24,c26,cchart RNN_GT_GEO_TAU=0.15 \
RNN_W_TEX=1 RNN_W_NONEDGE=3 RNN_MISS_VISIBLE=1 RNN_EDGE_HARD=2 \
EXTRA_TRAIN="cchartT0_train:3,cchartT1_train:2,cchartT2_train:2,cchartT3_train,cchartP0_train:2,cchartP1_train,cchartP2_train,cchartP3_train,c26g_train:3,c26p_train:2,c26z_train,c24g_train:2,c24p_train:2,c22txp_train:2,c20p_train" \
EXTRA_VAL="cchart0_val,cchart1_val,cchart2_val,cchart3_val,c26g_val,c26z_val,c26p_val,c24g_val,c22txp_val,c21txp_val,c20p_val" \
python train_v17.py --init runs/v22c/model.pt --base runs/v22c/model.pt --tag v27 --out runs/v27 --steps 6000 \
    --lr 3e-4 --eval_every 3000 --workers 2 > runs/v27/train.log 2>&1
echo TRAIN_OK
