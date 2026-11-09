#!/bin/sh
# cref v2 data (~30-35 min): repeated-pattern bursts (RNN_PATTERN=1, rnn/pattern_scene.py) + re-crop of the v1
# renders with the v2 tool (8 inputs, period slips, GCam labels from the median flow). Does NOT train.
SP=/c/Users/valen/AppData/Local/Temp/claude/b742f57fdf29e2e2/75d997e5-7f0e-41b3-bcd6-cbef711f0996/scratchpad
G=$SP/cref/gen; D=$SP/cref/data
export PYTHONDONTWRITEBYTECODE=1
export RNN_REAL="C:/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn/data/real"
export RNN_NFRAMES=8 RNN_TEX_SCALE=0.5 RNN_LEAN=1 RNN_SIZE=1024 RNN_ROT_SMALL=20,60 RNN_SMALL_DISP=0.1,3.0,0.5,0.7
t0=$(date +%s)
cd $G
# 5 render workers: 36 train pattern bursts + 4 val pattern bursts, interleaved.
NP=${NP:-36}; NPV=${NPV:-4}
for w in 0 1 2 3 4; do
    seeds=""; vseeds=""
    i=0; while [ $i -lt $NP ]; do [ $((i % 5)) -eq $w ] && seeds="$seeds $((500000 + i))"; i=$((i + 1)); done
    i=0; while [ $i -lt $NPV ]; do [ $((i % 5)) -eq $w ] && vseeds="$vseeds $((950000 + i))"; i=$((i + 1)); done
    ( RNN_PATTERN=1 python -B ../render_list.py "$(cygpath -w $D/rpP)" 0 $seeds
      [ -n "$vseeds" ] && RNN_PATTERN=1 python -B ../render_list.py "$(cygpath -w $D/rpPV)" 1 $vseeds ) > $SP/cref/render_w$w.log 2>&1 &
done
wait
echo "RENDER_OK pattern train $(ls $D/rpP/*.npz 2>/dev/null | wc -l) val $(ls $D/rpPV/*.npz 2>/dev/null | wc -l) $(( $(date +%s) - t0 ))s"
grep -h FAIL $SP/cref/render_w*.log
cd $SP/cref
rm -f $D/train/*.crops $D/val/*.crops
one() {   # $1 npz, $2 out dir, $3 seed
    b=$(basename $1 .npz)
    python -B npz2bin.py "$(cygpath -w $1)" "$(cygpath -w $2/$b.bin)"
    OMP_NUM_THREADS=2 ./cref_dataset.exe $2/$b.bin $2/$b.crops $3 12 80 > /dev/null 2>&1 || echo "CROP_FAIL $b"
    rm -f $2/$b.bin
}
i=0
for f in $D/rpA/*.npz $D/rpM/*.npz $D/rpP/*.npz; do
    i=$((i + 1)); one $f $D/train $i &
    [ $((i % 3)) -eq 0 ] && wait
done
wait
for f in $D/rpV/*.npz $D/rpPV/*.npz; do i=$((i + 1)); one $f $D/val $i & [ $((i % 3)) -eq 0 ] && wait; done
wait
echo "CROPS_OK train $(ls $D/train/*.crops | wc -l) val $(ls $D/val/*.crops | wc -l) $(( $(date +%s) - t0 ))s"
du -sh $D/train $D/val
python -B crops.py "$(cygpath -w $D/train)" 2>&1 | grep -v Warning
python -B crops.py "$(cygpath -w $D/val)" 2>&1 | grep -v Warning
echo GEN_DONE
