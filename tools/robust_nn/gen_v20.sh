#!/bin/sh
# v20 data: whole-camera rotation with the app's global homography, normal bursts with the homography, and
# perfectly alignable bursts (with / without the homography) to measure and train against over-rejection.
# Bursts render the four real scenes; caches run the app's stages through build/pipe2/tp_real20.exe.
#   sh gen_v20.sh > data/rp20_gen.log 2>&1
set -e
cd "$(dirname "$0")"
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
SC="APC_1186 APC_1294 APC_1340 APC_1342"
G="python real_protocol.py gen --scenes $SC --workers 6"
export RNN_ROT_PATCHES=2,10,1,5
RNN_BIGROT=3,8,150 $G --out data/rp20rot_train --n 240 --seed0 1300000
RNN_BIGROT=3,8,150 RNN_VAL_FROM_SCENES=1 $G --out data/rp20rot_val --n 30 --seed0 9300000 --val
$G --out data/rp20h_train --n 160 --seed0 1400000
RNN_VAL_FROM_SCENES=1 $G --out data/rp20h_val --n 30 --seed0 9400000 --val
RNN_PERFECT=1 RNN_BIGROT=3,8,150 $G --out data/rp20pH_train --n 80 --seed0 1500000
RNN_PERFECT=1 $G --out data/rp20pH_train --n 80 --seed0 1500100
RNN_PERFECT=1 RNN_BIGROT=3,8,150 RNN_VAL_FROM_SCENES=1 $G --out data/rp20pH_val --n 15 --seed0 9500000 --val
RNN_PERFECT=1 RNN_VAL_FROM_SCENES=1 $G --out data/rp20pH_val --n 15 --seed0 9500100 --val
RNN_PERFECT=still $G --out data/rp20p_train --n 160 --seed0 1600000
RNN_PERFECT=still RNN_VAL_FROM_SCENES=1 $G --out data/rp20p_val --n 30 --seed0 9600000 --val
echo GEN_OK
C="python precompute_cache.py --workers 5"
$C --src data/rp20rot_val --out data/c20rot_val --val --homog
$C --src data/rp20h_val --out data/c20h_val --val --homog
$C --src data/rp20pH_val --out data/c20pH_val --val --homog --no_corrupt
$C --src data/rp20p_val --out data/c20p_val --val --no_corrupt
$C --src data/rp20rot_train --out data/c20rot_train --variants 3 --corrupt_p 0.5 --homog
$C --src data/rp20h_train --out data/c20h_train --variants 3 --homog
$C --src data/rp20pH_train --out data/c20pH_train --variants 2 --homog --no_corrupt
$C --src data/rp20p_train --out data/c20p_train --variants 2 --no_corrupt
echo CACHE_OK
