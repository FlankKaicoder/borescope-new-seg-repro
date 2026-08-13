#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/borescope-new-seg-repro; O=$R/results/fast_repro/exp03_low_conf; mkdir -p "$R/results/fast_repro"
test ! -e "$O"
$R/.venv/bin/python $R/tools/evaluation/exp03_low_conf_fast_repro.py --weights $R/results/training/exp02_1_baseline_20260812T135254Z/artifacts/ultralytics/baseline/weights/best.pt --data-root /root/autodl-tmp/borescope-new-seg-data/v1 --manifest /root/autodl-tmp/borescope-new-seg-data/v1/split_manifest.csv --output "$O" --batch 32
