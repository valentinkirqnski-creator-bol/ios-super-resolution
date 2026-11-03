#!/bin/sh
# v28 training caches only, rebuilt with burst-consistent tile slips (TP_SMALL_CONSISTENT). Below-normal priority.
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn
sleep 2
exec > data/rp28_cache.log 2>&1
set -e
export RNN_TP_REAL="$(cygpath -w "$(pwd)/build/pipe2/tp_real20.exe")"
export TP_SMALL_RANGE=0.2,2.0 TP_SMALL_ANY=1 TP_SMALL_UNIFORM=1 TP_SMALL_CONSISTENT=1
for k in p t c; do
  python precompute_cache.py --workers 2 --oracle --src data/rp28${k}_train --out data/c28${k}_train --variants 2 --corrupt_p 0.8 --small
done
echo CACHE_OK
