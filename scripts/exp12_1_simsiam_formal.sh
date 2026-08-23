#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/borescope-new-seg-repro
RUN_ID="${EXP12_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="$ROOT/results/exp12_simsiam_basic/formal_${RUN_ID}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing Exp12.1 formal output: $OUT" >&2
  exit 2
fi

exec .venv/bin/python -m tools.ssl.exp12_simsiam_formal \
  --project-root "$ROOT" \
  --data-root /root/autodl-tmp/borescope-new-seg-data/v1 \
  --weights "$ROOT/weights/yolo11n-seg.pt" \
  --split-manifest "$ROOT/results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv" \
  --exp12-0-snapshot "$ROOT/results/ssl/exp12_0_encoder_analysis/parameter_snapshot_before.jsonl" \
  --smoke-summary "$ROOT/results/exp12_simsiam_basic/smoke_20260822T063541Z/summary.json" \
  --output "$OUT" \
  --imgsz 512 \
  --batch 32 \
  --workers 8 \
  --epochs 100 \
  --seed 42 \
  --lr 0.00625 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --checkpoint-interval 10
