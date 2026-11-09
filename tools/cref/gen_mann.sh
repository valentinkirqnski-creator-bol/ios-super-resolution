#!/bin/sh
# Misalignment-artifact NN data (~20-25 min): long-edge bursts (RNN_EDGE=1, rnn/edge_scene.py), half with a slight
# camera roll (tile stairs from the app's per-tile aligner), half with burst9's measured hand motion; then crops of
# ALL renders (general rpA, motion rpM, pattern rpP, edge rpE) with mann_dataset.exe (edge / stair / period / random
# slips, mann features, ghost labels, gate), 64 px crops, 10 per frame. Does NOT train.
SP=/c/Users/valen/AppData/Local/Temp/claude/b742f57fdf29e2e2/75d997e5-7f0e-41b3-bcd6-cbef711f0996/scratchpad
G=$SP/cref/gen; D=$SP/cref/data
export PYTHONDONTWRITEBYTECODE=1
export RNN_REAL="C:/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn/data/real"
export RNN_NFRAMES=8 RNN_TEX_SCALE=0.5 RNN_LEAN=1 RNN_SIZE=1024 RNN_ROT_SMALL=20,60 RNN_SMALL_DISP=0.1,3.0,0.5,0.7
MOT="$(cygpath -w $SP/wt_8a28f74/tools/robust_nn/data/motion_burst9.txt)"
[ -f "$SP/wt_8a28f74/tools/robust_nn/data/motion_burst9.txt" ] || { echo "NO MOTION FILE"; exit 1; }
t0=$(date +%s)
cd $G
NE=${NE:-40}; NEV=${NEV:-6}; NW=4
for w in 0 1 2 3; do
    roll=""; mot=""; vroll=""; vmot=""
    i=0; while [ $i -lt $NE ]; do
        if [ $((i % NW)) -eq $w ]; then if [ $((i % 2)) -eq 0 ]; then roll="$roll $((700000 + i))"; else mot="$mot $((700000 + i))"; fi; fi
        i=$((i + 1)); done
    i=0; while [ $i -lt $NEV ]; do
        if [ $((i % NW)) -eq $w ]; then if [ $((i % 2)) -eq 0 ]; then vroll="$vroll $((960000 + i))"; else vmot="$vmot $((960000 + i))"; fi; fi
        i=$((i + 1)); done
    ( [ -n "$roll" ] && RNN_EDGE=1 RNN_ROLL=0.3,1.5 python -B ../render_list.py "$(cygpath -w $D/rpE)" 0 $roll
      [ -n "$mot" ] && RNN_EDGE=1 RNN_MOTION="$MOT" python -B ../render_list.py "$(cygpath -w $D/rpE)" 0 $mot
      [ -n "$vroll" ] && RNN_EDGE=1 RNN_ROLL=0.3,1.5 python -B ../render_list.py "$(cygpath -w $D/rpEV)" 1 $vroll
      [ -n "$vmot" ] && RNN_EDGE=1 RNN_MOTION="$MOT" python -B ../render_list.py "$(cygpath -w $D/rpEV)" 1 $vmot
    ) > $SP/cref/mrender_w$w.log 2>&1 &
done
wait
echo "RENDER_OK edge train $(ls $D/rpE/*.npz 2>/dev/null | wc -l) val $(ls $D/rpEV/*.npz 2>/dev/null | wc -l) $(( $(date +%s) - t0 ))s"
grep -h FAIL $SP/cref/mrender_w*.log
cd $SP/cref
mkdir -p $D/mtrain $D/mval
rm -f $D/mtrain/*.crops $D/mval/*.crops
one() {   # $1 npz, $2 out dir, $3 seed
    b=$(basename $1 .npz)
    python -B npz2bin.py "$(cygpath -w $1)" "$(cygpath -w $2/$b.bin)"
    OMP_NUM_THREADS=2 ./mann_dataset.exe $2/$b.bin $2/$b.crops $3 10 64 > $2/$b.log 2>&1 || echo "CROP_FAIL $b"
    rm -f $2/$b.bin
}
i=0
for f in $D/rpA/*.npz $D/rpM/*.npz $D/rpP/*.npz $D/rpE/*.npz; do
    i=$((i + 1)); one $f $D/mtrain $i &
    [ $((i % 3)) -eq 0 ] && wait
done
wait
for f in $D/rpV/*.npz $D/rpPV/*.npz $D/rpEV/*.npz; do i=$((i + 1)); one $f $D/mval $i & [ $((i % 3)) -eq 0 ] && wait; done
wait
echo "CROPS_OK train $(ls $D/mtrain/*.crops | wc -l) val $(ls $D/mval/*.crops | wc -l) $(( $(date +%s) - t0 ))s"
du -sh $D/mtrain $D/mval
python -B -c "
import mcrops, numpy as np
for d in ('data/mtrain', 'data/mval'):
    a = mcrops.load_dir(d, 64); print(d, len(a))
"
python -B mcrops.py data/mval 64 2>&1 | head -4
echo GEN_DONE
