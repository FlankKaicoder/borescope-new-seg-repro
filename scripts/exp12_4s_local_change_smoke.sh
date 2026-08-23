#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/borescope-new-seg-repro
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/exp12_local_change/smoke_${RUN_ID}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing Exp12.4-S output: $OUT" >&2
  exit 2
fi

exec .venv/bin/python -m tools.ssl.exp12_local_change \
  --phase smoke \
  --project-root "$ROOT" \
  --data-root /root/autodl-tmp/borescope-new-seg-data/v1 \
  --weights "$ROOT/weights/yolo11n-seg.pt" \
  --split-manifest "$ROOT/results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv" \
  --exp12-0-snapshot "$ROOT/results/ssl/exp12_0_encoder_analysis/parameter_snapshot_before.jsonl" \
  --output "$OUT" \
  --imgsz 512 \
  --batch 16 \
  --workers 8 \
  --seed 42 \
  --lr 0.001 \
  --weight-decay 0.0001
