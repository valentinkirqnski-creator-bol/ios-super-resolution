#!/bin/sh
# v33 data: the burst12 misalignment, simulated exactly - a slow camera roll (0.3-0.6 deg/frame, up to ~3 deg by
# frame 7, +-0.3 deg spread) with a slow zoom (up to ~2.5%) and tremor, real photo scenes at camera scale, 8 frames -
# aligned by the app's own aligner WITHOUT the global homography (A: one translation per tile, long edges step at
# tile borders) and WITH it (H: clean). No deliberate corruption: the slips are the aligner's. Below-normal (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rp33_gen.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_NFRAMES=8 RNN_ROT_PATCHES=0,0,0,0 RNN_PERFECT=1 RNN_SIZE=1024 RNN_TEX_SCALE=0.5 RNN_B12=0.3,0.3,0.6,0.0015,0.004
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 2"
$G --out data/rp33_train --n 16 --seed0 3900000
RNN_VAL_FROM_SCENES=1 $G --out data/rp33_val --n 4 --seed0 9880000 --val
echo GEN_OK
C="python precompute_cache.py --workers 2 --no_corrupt"
$C --src data/rp33_val --out data/c33A_val --val
$C --src data/rp33_val --out data/c33H_val --val --homog
$C --src data/rp33_train --out data/c33A_train --variants 1
$C --src data/rp33_train --out data/c33H_train --variants 1 --homog
echo CACHE_OK
