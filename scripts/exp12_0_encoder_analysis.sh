#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/borescope-new-seg-repro
OUT="$ROOT/results/ssl/exp12_0_encoder_analysis"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing Exp12.0 output: $OUT" >&2
  exit 2
fi

.venv/bin/python -m tools.ssl.exp12_encoder_analysis \
  --project-root "$ROOT" \
  --data-root /root/autodl-tmp/borescope-new-seg-data/v1 \
  --weights "$ROOT/weights/yolo11n-seg.pt" \
  --output "$OUT" \
  --imgsz 512
