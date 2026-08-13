#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/borescope-new-seg-repro; D=/root/autodl-tmp/borescope-new-seg-data/v1; O=$R/results/final_verify/exp10; P=$R/.venv/bin/python
seed=${1:?seed 43 or 44}; [[ "$seed" == 43 || "$seed" == 44 ]]
$P $R/tools/training/exp10_train.py --mode baseline --seed "$seed" --model $R/weights/yolo11n-seg.pt --data $D/data.yaml --output $O/seed${seed}/baseline100
B=$O/seed${seed}/baseline100/ultralytics/baseline/weights/best.pt
$P $R/tools/evaluation/exp10_build_hard_pool.py --seed "$seed" --weights "$B" --data-root $D --manifest $D/split_manifest.csv --output $O/seed${seed}/hard_pool
$P $R/tools/training/exp10_train.py --mode control --seed "$seed" --model "$B" --data $D/data.yaml --hard-pool $O/seed${seed}/hard_pool/hard_pool.csv --output $O/seed${seed}/uniform_control30
$P $R/tools/training/exp10_train.py --mode treatment --seed "$seed" --model "$B" --data $D/data.yaml --hard-pool $O/seed${seed}/hard_pool/hard_pool.csv --output $O/seed${seed}/hard_treatment30
