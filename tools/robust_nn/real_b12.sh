#!/bin/sh
# burst12 (reference APC_1294): the NN mask WITHOUT the global homography, then WITH it. Masks + NN merge each.
# Waits for the running real_v32.sh job, builds a second exe (the first is in use) with the homography print.
set -e
cd "$(dirname "$0")"
MODEL=$1; TAG=$2
until grep -q RUN_OK data/real_v32.log 2>/dev/null; do sleep 15; done
ROOT="$(cd ../.. && pwd)"; B=build/pipe2/$TAG; H="$(pwd)/$B/w.h"
F="-O2 -std=gnu++17 -DHAVE_LIBRAW -I$ROOT/core -I$ROOT/tools/rob_refine -Icpp -I$ROOT/vendor/LibRaw -pthread -fopenmp"
CORE="$(for o in align grey_pyramid host_stubs kernels merge raw_io robustness snr_tuning global_homography; do printf 'build/pipe2/%s.o ' $o; done)"
g++ $F "-DRNN_WEIGHTS_HEADER=\"$H\"" -c cpp/burst_compare.cpp -o $B/burst_compare2.o
g++ $F -o $B/burst_compare2.exe $B/burst_compare2.o $B/robust_nn.o $CORE $ROOT/vendor/LibRaw/lib/libraw.a -lws2_32 -static
echo BUILD2_OK
for HM in noH H; do
    O=data/real_cmp/${TAG}_burst12_$HM; mkdir -p $O
    unset BC_HOMOG; [ $HM = H ] && export BC_HOMOG=1
    BC_RESID_FEAT=1 BC_GAIN_NORM=1 BC_MERGES=nn ./$B/burst_compare2.exe $O 0 0 4032 3024 1.0 0.2 0.9 \
        /c/Users/valen/Downloads/burst12/APC_1294.dng $(ls /c/Users/valen/Downloads/burst12/*.dng | grep -v APC_1294) 2> $O/log.txt
    echo "$HM: $(grep 'mean R' $O/log.txt | sed 's/.*mean R //' | tr '\n' ' ')"
    grep "homography rotation" $O/log.txt | head -8
done
echo B12_OK
