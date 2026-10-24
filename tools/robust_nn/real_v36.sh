#!/bin/sh
# v36 on the real bursts: burst12 (ref APC_1294) WITHOUT the homography, burst9 (steady), burst11 WITH the homography;
# NN merge + masks each.   sh real_v36.sh runs/v36/model_s3522.pt v36
set -e
cd "$(dirname "$0")"
MODEL=$1; TAG=$2
ROOT="$(cd ../.. && pwd)"; B=build/pipe2/$TAG; mkdir -p $B
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
run() {   # name, homography flag, dng list...
    N=$1; HM=$2; shift 2; O=data/real_cmp/${TAG}_$N; mkdir -p $O
    unset BC_HOMOG; [ "$HM" = H ] && export BC_HOMOG=1
    BC_RESID_FEAT=1 BC_GAIN_NORM=1 BC_MERGES=nn ./$B/burst_compare.exe $O 0 0 4032 3024 1.0 0.2 0.9 "$@" 2> $O/log.txt
    echo "$N: $(grep 'mean R' $O/log.txt | sed 's/.*mean R //' | tr '\n' ' ')"
}
run burst12_noH noH /c/Users/valen/Downloads/burst12/APC_1294.dng $(ls /c/Users/valen/Downloads/burst12/*.dng | grep -v APC_1294)
run burst9_nn noH $(ls /c/Users/valen/Downloads/burst9/*.dng)
run burst11_nn H $(ls /c/Users/valen/Downloads/burst11/*.dng)
echo RUN_OK
