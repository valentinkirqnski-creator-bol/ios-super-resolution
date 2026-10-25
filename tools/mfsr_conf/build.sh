#!/bin/sh
# Builds the C++ harness against the CURRENT core/ sources. Fresh objects in build/: an object compiled against an
# older types.h silently corrupts Config (see the MinGW harness notes). Same -std for every TU, static link.
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
B=build
mkdir -p $B
F="-O2 -std=gnu++17 -DHAVE_LIBRAW -I$ROOT/core -I$ROOT/tools/rob_refine -I$ROOT/vendor/LibRaw -pthread -fopenmp"
for s in align grey_pyramid kernels merge raw_io robustness snr_tuning global_homography; do
    if [ ! "$B/$s.o" -nt "$ROOT/core/$s.cpp" ] || [ ! "$B/$s.o" -nt "$ROOT/core/types.h" ]; then
        g++ $F -c "$ROOT/core/$s.cpp" -o "$B/$s.o"
    fi
done
[ "$B/host_stubs.o" -nt "$ROOT/tools/rob_refine/host_stubs.cpp" ] || g++ $F -c "$ROOT/tools/rob_refine/host_stubs.cpp" -o "$B/host_stubs.o"
g++ $F -c cpp/conf_harness.cpp -o $B/conf_harness.o
g++ $F -o $B/conf_harness.exe $B/conf_harness.o $B/align.o $B/grey_pyramid.o $B/kernels.o $B/merge.o $B/raw_io.o \
    $B/robustness.o $B/snr_tuning.o $B/global_homography.o $B/host_stubs.o "$ROOT/vendor/LibRaw/lib/libraw.a" -lws2_32 -static
echo BUILD_OK
# build line for frame_dump (real-burst evaluation)
g++ $F -c cpp/frame_dump.cpp -o $B/frame_dump.o && g++ $F -o $B/frame_dump.exe $B/frame_dump.o $B/raw_io.o $B/host_stubs.o "$ROOT/vendor/LibRaw/lib/libraw.a" -lws2_32 -static
