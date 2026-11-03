#!/bin/sh
# v38 data: FULL 12 MP (4032 x 3024) 8-frame bursts of burst9's scene at camera scale, three twins per burst with the
# same crop and motion: S = still camera, no patches (perfect); M = burst9's measured frame-to-frame motion, no
# patches (the roll residual only); C = that motion + 40-80 content patches translated / rotated by 0.1-3 px (small to
# big) + tile-vector slips of 0.1-3 px at cache time (half burst-consistent). All with EXACT per-tile translation
# (oracle) so the only misalignments are the deliberate ones + the in-tile roll residual. Stored as 384-guide-px tiles
# (768 raw) with their own source tiles. Below-normal priority (caller). ~hours.
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rp38_gen.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_NFRAMES=8 RNN_ROT_PATCHES=0,0,0,0 RNN_SIZE=${SZ_H:-3024} RNN_SIZE_W=${SZ_W:-4032} RNN_TEX_SCALE=0.5 RNN_LEAN=1
G="python real_protocol.py gen --scenes APC_1186 --workers ${WORKERS:-2}"
NTR=${NTR:-6}; NVA=${NVA:-2}
# S: still, perfect
env RNN_PERFECT=1 RNN_MOTION="$(pwd)/data/motion_zero.txt" $G --out data/rp38S_train --n $NTR --seed0 4100000
env RNN_PERFECT=1 RNN_MOTION="$(pwd)/data/motion_zero.txt" RNN_VAL_FROM_SCENES=1 $G --out data/rp38S_val --n $NVA --seed0 9900000 --val
# M: burst9's motion, no patches
env RNN_PERFECT=1 RNN_MOTION="$(pwd)/data/motion_burst9.txt" $G --out data/rp38M_train --n $NTR --seed0 4100000
env RNN_PERFECT=1 RNN_MOTION="$(pwd)/data/motion_burst9.txt" RNN_VAL_FROM_SCENES=1 $G --out data/rp38M_val --n $NVA --seed0 9900000 --val
# C: burst9's motion + content patches (small to big)
env RNN_MOTION="$(pwd)/data/motion_burst9.txt" RNN_ROT_SMALL=40,80 RNN_SMALL_DISP=0.1,3.0,0.5,0.7 $G --out data/rp38C_train --n $NTR --seed0 4100000
env RNN_MOTION="$(pwd)/data/motion_burst9.txt" RNN_ROT_SMALL=40,80 RNN_SMALL_DISP=0.1,3.0,0.5,0.7 RNN_VAL_FROM_SCENES=1 $G --out data/rp38C_val --n $NVA --seed0 9900000 --val
echo GEN_OK
T=${TILE:-384}
C="python precompute_cache.py --workers ${WORKERS:-2} --oracle --tile_out $T"
$C --src data/rp38S_val --out data/c38S_val --val --no_corrupt
$C --src data/rp38M_val --out data/c38M_val --val --no_corrupt
TP_SMALL_RANGE=0.1,3.0 TP_SMALL_ANY=1 TP_SMALL_UNIFORM=1 TP_SMALL_CONSISTENT=1 $C --src data/rp38C_val --out data/c38C_val --val --val_corrupt_only --small
$C --src data/rp38S_train --out data/c38S_train --variants 1 --no_corrupt
$C --src data/rp38M_train --out data/c38M_train --variants 1 --no_corrupt
TP_SMALL_RANGE=0.1,3.0 TP_SMALL_ANY=1 TP_SMALL_UNIFORM=1 TP_SMALL_CONSISTENT=1 $C --src data/rp38C_train --out data/c38C_train --variants 1 --corrupt_p 1.0 --small
echo CACHE_OK
