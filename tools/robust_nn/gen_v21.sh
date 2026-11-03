#!/bin/sh
# v21 data: aliased fine-text bursts (rnn/text_scene.py), 8 frames of pure sub-pixel translation. Each burst is
# cached twice through the app's stages with PERFECT alignment (tp_real oracle flag): as is (txp: every visible
# pixel should be kept) and with deliberate misalignment on the same text (txm: random / edge-targeted tile errors
# and straight-edge sub-pixel steps), so aliasing and misalignment are seen on identical content.
#   sh gen_v21.sh > data/rp21_gen.log 2>&1
set -e
cd "$(dirname "$0")"
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export RNN_TEXT=1
G="python real_protocol.py gen --scenes APC_1186 APC_1294 APC_1340 APC_1342 --workers 6"
$G --out data/rp21tx_train --n 300 --seed0 2200000
$G --out data/rp21tx_val --n 30 --seed0 9700000 --val
echo GEN_OK
C="python precompute_cache.py --workers 5 --oracle"
$C --src data/rp21tx_val --out data/c21txp_val --val --no_corrupt
$C --src data/rp21tx_val --out data/c21txm_val --val --val_corrupt_only --edge_step
$C --src data/rp21tx_train --out data/c21txp_train --variants 2 --no_corrupt
$C --src data/rp21tx_train --out data/c21txm_train --variants 2 --corrupt_p 1.0 --edge_step
echo CACHE_OK
