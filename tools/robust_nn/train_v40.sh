#!/bin/sh
# v40: v21 widened with the motion-prior / structure-tensor inputs (rnn/features.py motion_structure_features), new
# inputs zero-weighted so step 0 IS v21, fine-tuned on v21's mix + the photo twins. GT is stored in the caches
# (gen_v40.sh), so RNN_GT_NMERGE must NOT be set here.
#   sh train_v40.sh [tag] [MOTF_KEEP]     tag v40 (all 10), v40b (7: no burst terms), v40c (4: motion prior only)
set -e
cd "$(dirname "$0")"
TAG=${1:-v40}
export MOTF=1 MOTF_KEEP=${2:-10} RNN_W_PERFECT=1
unset RNN_GT_NMERGE
export EXTRA_TRAIN="c20rot_train,c20h_train,c20pH_train,c20p_train,c21txp_train:2,c21txm_train:2,c40P_train,c40M_train,c40S_train,c40A_train"
export EXTRA_VAL="c20rot_val,c20p_val,c21txp_val,c21txm_val,c40P_val,c40M_val,c40S_val,c40A_val"
python train_v17.py --init runs/v21/model.pt --base runs/v21/model.pt --widen --tag $TAG --out runs/$TAG \
    --steps 10000 --eval_every 2500 --new_weight 1 --workers 3 --skip_s0 > runs/${TAG}_train.log 2>&1
echo TRAIN_EXIT $? >> runs/${TAG}_train.log
