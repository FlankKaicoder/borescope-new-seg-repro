#!/usr/bin/env bash
set -euo pipefail

R=/root/autodl-tmp/borescope-new-seg-repro
D=/root/autodl-tmp/borescope-new-seg-data/v1
P=$R/.venv/bin/python
BASE=$R/results/final_verify/exp10/seed43/baseline100/ultralytics/baseline/weights/best.pt
POOL=$R/results/final_verify/exp10/seed43/hard_pool/hard_pool.csv
OUT=$R/results/final_verify/exp10_controlled_restart/seed43/hard_treatment30
MODEL_SHA=09f115c80de4d624f4fb36ee8ede1a65a372cce395d963a8e02e4b6bd65e732c
POOL_SHA=645ef8731c721b5dbdb9da596e839185db6c6c72021acc7f8caa56b878e8f6d9

[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }

"$P" "$R/tools/training/exp10_controlled_run.py" \
  --action preflight --mode treatment --seed 43 \
  --model "$BASE" --expected-model-sha256 "$MODEL_SHA" \
  --data "$D/data.yaml" --data-root "$D" \
  --hard-pool "$POOL" --expected-hard-pool-sha256 "$POOL_SHA" \
  --output "$OUT/preflight"

"$P" "$R/tools/training/exp10_controlled_run.py" \
  --action train --mode treatment --seed 43 \
  --model "$BASE" --expected-model-sha256 "$MODEL_SHA" \
  --data "$D/data.yaml" --data-root "$D" \
  --hard-pool "$POOL" --expected-hard-pool-sha256 "$POOL_SHA" \
  --preflight-summary "$OUT/preflight/preflight_summary.json" \
  --output "$OUT/formal"
