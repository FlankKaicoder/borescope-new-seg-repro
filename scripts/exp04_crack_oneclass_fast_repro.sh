#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/borescope-new-seg-repro; D=/root/autodl-tmp/borescope-new-seg-data/crack_only_v1; O=$R/results/fast_repro/exp04_crack_oneclass
test ! -e "$D"; test ! -e "$O"
$R/.venv/bin/python $R/tools/dataset/build_crack_only.py --source /root/autodl-tmp/borescope-new-seg-data/v1 --manifest /root/autodl-tmp/borescope-new-seg-data/v1/split_manifest.csv --output "$D"
$R/.venv/bin/python $R/tools/training/fasttrack_train.py --model $R/weights/yolo11n-seg.pt --data $D/data.yaml --output "$O/train" --name crack_only --epochs 100 --batch 32
$R/.venv/bin/python $R/tools/evaluation/evaluate_seg_model.py --weights "$O/train/ultralytics/crack_only/weights/best.pt" --data $D/data.yaml --split val --imgsz 640 --conf .001 --iou .70 --batch 32 --output "$O/eval"
