#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/borescope-new-seg-repro; O=$R/results/fast_repro/exp05_hard_mining; B=$R/results/training/exp02_1_baseline_20260812T135254Z/artifacts/ultralytics/baseline/weights/best.pt; D=/root/autodl-tmp/borescope-new-seg-data/v1/data.yaml
test ! -e "$O"; mkdir -p "$O"
$R/.venv/bin/python $R/tools/evaluation/exp05_build_hard_pool.py --weights "$B" --data-root /root/autodl-tmp/borescope-new-seg-data/v1 --manifest /root/autodl-tmp/borescope-new-seg-data/v1/split_manifest.csv --output "$O/hard_pool" --batch 32
$R/.venv/bin/python $R/tools/training/exp05_sampler_train.py --mode control --model "$B" --data "$D" --hard-pool "$O/hard_pool/hard_pool.csv" --output "$O/control"
$R/.venv/bin/python $R/tools/training/exp05_sampler_train.py --mode treatment --model "$B" --data "$D" --hard-pool "$O/hard_pool/hard_pool.csv" --output "$O/treatment"
$R/.venv/bin/python $R/tools/evaluation/evaluate_seg_model.py --weights "$O/control/ultralytics/control/weights/best.pt" --data "$D" --split val --batch 32 --output "$O/control_eval"
$R/.venv/bin/python $R/tools/evaluation/evaluate_seg_model.py --weights "$O/treatment/ultralytics/treatment/weights/best.pt" --data "$D" --split val --batch 32 --output "$O/treatment_eval"
$R/.venv/bin/python $R/tools/evaluation/fixed_point_errors.py --weights "$O/control/ultralytics/control/weights/best.pt" --data-root /root/autodl-tmp/borescope-new-seg-data/v1 --manifest /root/autodl-tmp/borescope-new-seg-data/v1/split_manifest.csv --output "$O/control_eval/fixed_errors.json"
$R/.venv/bin/python $R/tools/evaluation/fixed_point_errors.py --weights "$O/treatment/ultralytics/treatment/weights/best.pt" --data-root /root/autodl-tmp/borescope-new-seg-data/v1 --manifest /root/autodl-tmp/borescope-new-seg-data/v1/split_manifest.csv --output "$O/treatment_eval/fixed_errors.json"
