#!/bin/sh
# v40 data: v21's training mix rebuilt (the caches were deleted 2026-10-27) PLUS photo-scene twins.
#   anchors   c15 (APC_1186 + HDRP_N000, val APC_1211), c17 (APC_1294/1340/1342) - corrupted + app-aligned variants
#   rotation  c20rot / c20h / c20pH / c20p (gen_v20 recipe, the app's homography path)
#   text      c21txp / c21txm (gen_v21 recipe, same seeds): aliased fine text, perfect vs deliberately misaligned
#   twins     c40P / c40M / c40S / c40A: the SAME photo bursts (pure hand-shake translation, 8 frames) perfectly
#             aligned (oracle), oracle + deliberate tile misalignment (random / block / edge tiles / edge steps),
#             oracle + tiny edge slips only, and the app's own aligner. Train at 2 texels/px and camera scale; val on
#             the held-out scene APC_1211 at camera scale.
#   sh gen_v40.sh > data/rp40_gen.log 2>&1
set -e
cd "$(dirname "$0")"
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_LEAN=1
# the merged-noise ground truth v21 was trained against, stored IN the caches (identical to the loader's recompute up
# to fp16), so the source bursts can be deleted once cached: train WITHOUT RNN_GT_NMERGE (no recompute)
export RNN_GT_NMERGE=8 RNN_CACHE_COMPRESS=1
G="python real_protocol.py gen --workers 6"
C="python precompute_cache.py --workers 5"

# ---- anchors (c15 / c17) ----
export RNN_ROT_PATCHES=2,10,1,5
$G --scenes APC_1186 --out data/rp13_train --n 300 --seed0 3100000
$G --scenes HDRP_N000 --out data/rp13_train_hdrp --n 100 --seed0 3150000
$G --scenes APC_1186 --out data/rp13_val --n 30 --seed0 9100000 --val
$G --scenes APC_1294 APC_1340 APC_1342 --out data/rp17_train --n 300 --seed0 3200000
RNN_VAL_FROM_SCENES=1 $G --scenes APC_1294 APC_1340 APC_1342 --out data/rp17_val --n 30 --seed0 9200000 --val
echo GEN_ANCHOR_OK
$C --src data/rp13_val --out data/c15_val --val
$C --src data/rp17_val --out data/c17_val --val
$C --src data/rp13_train,data/rp13_train_hdrp --out data/c15_train --variants 3
$C --src data/rp17_train --out data/c17_train --variants 3
echo CACHE_ANCHOR_OK
rm -rf data/rp13_train data/rp13_train_hdrp data/rp13_val data/rp17_train data/rp17_val

# ---- rotation / homography (gen_v20 recipe, 2 variants) ----
SC="APC_1186 APC_1294 APC_1340 APC_1342"
G20="$G --scenes $SC"
RNN_BIGROT=3,8,150 $G20 --out data/rp20rot_train --n 240 --seed0 1300000
RNN_BIGROT=3,8,150 RNN_VAL_FROM_SCENES=1 $G20 --out data/rp20rot_val --n 30 --seed0 9300000 --val
$G20 --out data/rp20h_train --n 160 --seed0 1400000
RNN_PERFECT=1 RNN_BIGROT=3,8,150 $G20 --out data/rp20pH_train --n 80 --seed0 1500000
RNN_PERFECT=1 $G20 --out data/rp20pH_train --n 80 --seed0 1500100
RNN_PERFECT=still $G20 --out data/rp20p_train --n 160 --seed0 1600000
RNN_PERFECT=still RNN_VAL_FROM_SCENES=1 $G20 --out data/rp20p_val --n 30 --seed0 9600000 --val
echo GEN_ROT_OK
$C --src data/rp20rot_val --out data/c20rot_val --val --homog
$C --src data/rp20p_val --out data/c20p_val --val --no_corrupt
$C --src data/rp20rot_train --out data/c20rot_train --variants 2 --corrupt_p 0.5 --homog
$C --src data/rp20h_train --out data/c20h_train --variants 2 --homog
$C --src data/rp20pH_train --out data/c20pH_train --variants 2 --homog --no_corrupt
$C --src data/rp20p_train --out data/c20p_train --variants 2 --no_corrupt
echo CACHE_ROT_OK
rm -rf data/rp20rot_train data/rp20rot_val data/rp20h_train data/rp20pH_train data/rp20p_train data/rp20p_val
unset RNN_ROT_PATCHES

# ---- text twins (gen_v21 recipe, same seeds) ----
T="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 6"
RNN_TEXT=1 $T --out data/rp21tx_train --n 300 --seed0 2200000
RNN_TEXT=1 $T --out data/rp21tx_val --n 30 --seed0 9700000 --val
echo GEN_TEXT_OK
CO="$C --oracle"
RNN_TEXT=1 $CO --src data/rp21tx_val --out data/c21txp_val --val --no_corrupt
RNN_TEXT=1 $CO --src data/rp21tx_val --out data/c21txm_val --val --val_corrupt_only --edge_step
RNN_TEXT=1 $CO --src data/rp21tx_train --out data/c21txp_train --variants 2 --no_corrupt
RNN_TEXT=1 $CO --src data/rp21tx_train --out data/c21txm_train --variants 2 --corrupt_p 1.0 --edge_step
echo CACHE_TEXT_OK
rm -rf data/rp21tx_train data/rp21tx_val

# ---- v40 photo twins ----
S40="APC_1186 APC_1258 APC_1278 APC_1294 APC_1340 APC_1342 HDRP_N000"
export RNN_PERFECT=still RNN_NFRAMES=8
$G --scenes $S40 --out data/rp40ph_train --n 200 --seed0 4000000
RNN_TEX_SCALE=0.5 $G --scenes $S40 --out data/rp40pc_train --n 200 --seed0 4100000
RNN_TEX_SCALE=0.5 $G --scenes $S40 --out data/rp40pc_val --n 30 --seed0 9400400 --val
echo GEN_TWIN_OK
TW=data/rp40ph_train,data/rp40pc_train
$C --src data/rp40pc_val --out data/c40P_val --val --oracle --no_corrupt
$C --src data/rp40pc_val --out data/c40M_val --val --oracle --val_corrupt_only --edge_step
$C --src data/rp40pc_val --out data/c40S_val --val --oracle --val_corrupt_only --small
$C --src data/rp40pc_val --out data/c40A_val --val
$C --src $TW --out data/c40P_train --variants 1 --oracle --no_corrupt
$C --src $TW --out data/c40M_train --variants 2 --oracle --corrupt_p 1.0 --edge_step
$C --src $TW --out data/c40S_train --variants 1 --oracle --corrupt_p 1.0 --small
$C --src $TW --out data/c40A_train --variants 1 --corrupt_p 0.3
echo CACHE_TWIN_OK
rm -rf data/rp40ph_train data/rp40pc_train data/rp40pc_val
echo ALL_OK
