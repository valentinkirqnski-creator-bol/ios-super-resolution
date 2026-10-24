#!/bin/sh
# v37 data: burst9's own scene (APC_1186) under burst9's MEASURED frame-to-frame camera motion (data/motion_burst9.txt,
# replayed with 0.7-1.3x gain and random rotation sign), camera scale, 8 frames, no deliberate corruption. Three twins
# per burst: O = exact per-tile translation (perfect), A = the app's aligner without the homography (the in-tile
# rotation + translation residual it cannot follow: the car-plate edges), H = with the homography. Below-normal.
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rpb9_gen.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_NFRAMES=8 RNN_ROT_PATCHES=0,0,0,0 RNN_PERFECT=1 RNN_SIZE=1024 RNN_TEX_SCALE=0.5 RNN_MOTION="$(pwd)/data/motion_burst9.txt"
G="python real_protocol.py gen --scenes APC_1186 --workers 2"
$G --out data/rpb9_train --n 20 --seed0 4000000
RNN_VAL_FROM_SCENES=1 $G --out data/rpb9_val --n 4 --seed0 9890000 --val
echo GEN_OK
C="python precompute_cache.py --workers 2 --no_corrupt"
$C --src data/rpb9_val --out data/cb9O_val --val --oracle
$C --src data/rpb9_val --out data/cb9A_val --val
$C --src data/rpb9_val --out data/cb9H_val --val --homog
$C --src data/rpb9_train --out data/cb9O_train --variants 1 --oracle
$C --src data/rpb9_train --out data/cb9A_train --variants 1
$C --src data/rpb9_train --out data/cb9H_train --variants 1 --homog
echo CACHE_OK
