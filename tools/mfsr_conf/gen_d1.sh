#!/bin/sh
# Dataset d1 (configs/data_d1.yaml): val and test first (evaluation sets exist early), then train.
# Caller runs this at below-normal priority; 2 workers x 2 threads.
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd "$(dirname "$0")"
mkdir -p data/d1
python -W ignore gen_data.py configs/data_d1.yaml --splits val test train >> data/d1/gen.log 2>&1
echo GEN_DONE >> data/d1/gen.log
