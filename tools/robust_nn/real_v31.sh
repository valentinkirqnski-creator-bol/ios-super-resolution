#!/bin/sh
# burst11 (reference APC_1278) through the app's CPU stages with global homography: the analytic mask and the
# v31 NN mask, full frame, masks + both merges (two runs: one merge each, to fit in memory).
#   sh real_v31.sh runs/v31/model_s5257.pt v31
set -e
cd "$(dirname "$0")"
MODEL=$1; TAG=$2
ROOT="$(cd ../.. && pwd)"
B=build/pipe2/$TAG
mkdir -p $B
python - "$MODEL" "$B/w.h" <<'PY'
import sys
sys.path.insert(0, ".")
from real_protocol import load_net
from export_weights import write_header
write_header(load_net(sys.argv[1]), sys.argv[2], f"{sys.argv[1]}")
PY
F="-O2 -std=gnu++17 -DHAVE_LIBRAW -I$ROOT/core -I$ROOT/tools/rob_refine -Icpp -I$ROOT/vendor/LibRaw -pthread -fopenmp"
H="$(pwd)/$B/w.h"
CORE="$(for o in align grey_pyramid host_stubs kernels merge raw_io robustness snr_tuning global_homography; do printf 'build/pipe2/%s.o ' $o; done)"
g++ $F "-DRNN_WEIGHTS_HEADER=\"$H\"" -c cpp/robust_nn.cpp -o $B/robust_nn.o
g++ $F "-DRNN_WEIGHTS_HEADER=\"$H\"" -c cpp/burst_compare.cpp -o $B/burst_compare.o
g++ $F -o $B/burst_compare.exe $B/burst_compare.o $B/robust_nn.o $CORE $ROOT/vendor/LibRaw/lib/libraw.a -lws2_32 -static
echo BUILD_OK
for M in nn r0; do
    O=data/real_cmp/${TAG}_burst11_$M; mkdir -p $O
    BC_HOMOG=1 BC_RESID_FEAT=1 BC_GAIN_NORM=1 BC_MERGES=$M ./$B/burst_compare.exe $O 0 0 4032 3024 1.0 0.2 0.9 \
        $(ls /c/Users/valen/Downloads/burst11/*.dng) 2> $O/log.txt
    echo "$M: $(grep 'mean R' $O/log.txt | sed 's/.*mean R //' | tr '\n' ' ')"
done
echo RUN_OK
