#!/bin/sh
# v22 data: tiny misalignments (the slips that thicken / double an edge) vs well-aligned aliasing. Every burst is
# cached with PERFECT alignment (tp_real oracle flag) so the only misalignments are the deliberate tiny ones:
#   tx  : aliased text, capitals 1.8-9 raw px, 8 frames of sub-pixel translation
#         -> c22txp (perfectly aligned) and c22txm (tiny edge slips: 0.05-0.5 px tile shifts + straight-edge steps)
#   txr : the same kind of text with tiny tilt / shift patches on edges (translate, rotate or both;
#         0.1-1.5 deg, 0.03-0.4 px), half also with the tiny tile slips -> c22txr
#   scr : real photo scenes, translation-only camera, the same tiny patches and slips -> c22scr
#   sh gen_v22.sh > data/rp22_gen.log 2>&1
set -e
cd "$(dirname "$0")"
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 6"
TINY="RNN_ROT_SMALL=8,20 RNN_SMALL_MODE=0.1,1.5,0.03,0.4"
env RNN_TEXT=1 RNN_TEXT_CAP=1.8,9 $G --out data/rp22tx_train --n 240 --seed0 2400000
env RNN_TEXT=1 RNN_TEXT_CAP=1.8,9 $G --out data/rp22tx_val --n 30 --seed0 9800000 --val
env RNN_TEXT=1 RNN_TEXT_CAP=1.8,9 $TINY $G --out data/rp22txr_train --n 200 --seed0 2500000
env RNN_TEXT=1 RNN_TEXT_CAP=1.8,9 $TINY $G --out data/rp22txr_val --n 30 --seed0 9810000 --val
env RNN_PERFECT=still RNN_ROT_PATCHES=0,0,0,0 $TINY $G --out data/rp22scr_train --n 200 --seed0 2600000
env RNN_PERFECT=still RNN_ROT_PATCHES=0,0,0,0 $TINY $G --out data/rp22scr_val --n 30 --seed0 9820000 --val
echo GEN_OK
C="python precompute_cache.py --workers 5 --oracle"
$C --src data/rp22tx_val --out data/c22txp_val --val --no_corrupt
$C --src data/rp22tx_val --out data/c22txm_val --val --val_corrupt_only --small --edge_step
$C --src data/rp22txr_val --out data/c22txr_val --val --small --edge_step
$C --src data/rp22scr_val --out data/c22scr_val --val --small --edge_step
$C --src data/rp22tx_train --out data/c22txp_train --variants 2 --no_corrupt
$C --src data/rp22tx_train --out data/c22txm_train --variants 2 --corrupt_p 1.0 --small --edge_step
$C --src data/rp22txr_train --out data/c22txr_train --variants 2 --corrupt_p 0.5 --small --edge_step
$C --src data/rp22scr_train --out data/c22scr_train --variants 2 --corrupt_p 0.5 --small --edge_step
echo CACHE_OK
