#!/bin/sh
# v32 data: PAIRS of the same burst - perfectly aligned (the app's own aligner + global homography, no deliberate
# corruption) and with small slips (0.2-2 px tile vectors, half burst-consistent, + content patches) - including
# whole-camera rotations (RNN_BIGROT, up to ~30 deg, the burst11 case) at camera scale, so the net learns that a
# correctly aligned rotated frame is NOT a slip. Text and chart pairs too. Below-normal priority (caller).
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rp32_gen.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_NFRAMES=8 RNN_ROT_PATCHES=0,0,0,0 RNN_ROT_SMALL=6,14 RNN_SMALL_DISP=0.2,2.0,0.5,0.7
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 2"
# whole-camera rotation, perfectly alignable scene, camera scale (the content patches are the only slips)
env RNN_SIZE=1024 RNN_PERFECT=1 RNN_BIGROT=3,8,150 RNN_TEX_SCALE=0.5 $G --out data/rp32rot_train --n 12 --seed0 3800000
env RNN_SIZE=1024 RNN_PERFECT=1 RNN_BIGROT=3,8,150 RNN_TEX_SCALE=0.5 RNN_VAL_FROM_SCENES=1 $G --out data/rp32rot_val --n 3 --seed0 9860000 --val
# small roll only (0.5-4 deg), camera scale
env RNN_SIZE=1024 RNN_PERFECT=1 RNN_BIGROT=1,4,150 RNN_TEX_SCALE=0.5 $G --out data/rp32roll_train --n 8 --seed0 3810000
env RNN_SIZE=1024 RNN_PERFECT=1 RNN_BIGROT=1,4,150 RNN_TEX_SCALE=0.5 RNN_VAL_FROM_SCENES=1 $G --out data/rp32roll_val --n 2 --seed0 9870000 --val
echo GEN_OK
C="python precompute_cache.py --workers 2 --homog"
export TP_SMALL_RANGE=0.2,2.0 TP_SMALL_ANY=1 TP_SMALL_UNIFORM=1 TP_SMALL_CONSISTENT=1
for k in rot roll; do
  $C --src data/rp32${k}_val --out data/c32${k}P_val --val --no_corrupt
  $C --src data/rp32${k}_val --out data/c32${k}C_val --val --val_corrupt_only --small
  $C --src data/rp32${k}_train --out data/c32${k}P_train --variants 1 --no_corrupt
  $C --src data/rp32${k}_train --out data/c32${k}C_train --variants 1 --corrupt_p 1.0 --small
done
echo CACHE_OK
