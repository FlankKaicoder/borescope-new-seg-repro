#!/usr/bin/env bash
set -euo pipefail

R=/root/autodl-tmp/borescope-new-seg-repro
D=/root/autodl-tmp/borescope-new-seg-data/v1
P=$R/.venv/bin/python
RUN=$R/tools/training/exp10_controlled_run.py
ROOT=$R/results/final_verify/exp10_controlled_restart/seed44
OFFICIAL=$R/weights/yolo11n-seg.pt
OFFICIAL_SHA=55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152

[[ ! -e "$ROOT" ]] || { echo "Refusing to overwrite $ROOT" >&2; exit 2; }

"$P" "$RUN" \
  --action preflight --mode baseline --seed 44 \
  --model "$OFFICIAL" --expected-model-sha256 "$OFFICIAL_SHA" \
  --data "$D/data.yaml" --data-root "$D" \
  --output "$ROOT/baseline100/preflight"

"$P" "$RUN" \
  --action train --mode baseline --seed 44 \
  --model "$OFFICIAL" --expected-model-sha256 "$OFFICIAL_SHA" \
  --data "$D/data.yaml" --data-root "$D" \
  --preflight-summary "$ROOT/baseline100/preflight/preflight_summary.json" \
  --output "$ROOT/baseline100/formal"

BASE=$ROOT/baseline100/formal/ultralytics/baseline/weights/best.pt
BASE_SHA=$(sha256sum "$BASE" | awk '{print $1}')

"$P" "$R/tools/evaluation/exp10_build_hard_pool.py" \
  --seed 44 --weights "$BASE" --data-root "$D" \
  --manifest "$D/split_manifest.csv" --output "$ROOT/hard_pool"

POOL=$ROOT/hard_pool/hard_pool.csv
POOL_SHA=$(sha256sum "$POOL" | awk '{print $1}')

for MODE in control treatment; do
  if [[ "$MODE" == control ]]; then NAME=uniform_control30; else NAME=hard_treatment30; fi
  "$P" "$RUN" \
    --action preflight --mode "$MODE" --seed 44 \
    --model "$BASE" --expected-model-sha256 "$BASE_SHA" \
    --data "$D/data.yaml" --data-root "$D" \
    --hard-pool "$POOL" --expected-hard-pool-sha256 "$POOL_SHA" \
    --output "$ROOT/$NAME/preflight"

  "$P" "$RUN" \
    --action train --mode "$MODE" --seed 44 \
    --model "$BASE" --expected-model-sha256 "$BASE_SHA" \
    --data "$D/data.yaml" --data-root "$D" \
    --hard-pool "$POOL" --expected-hard-pool-sha256 "$POOL_SHA" \
    --preflight-summary "$ROOT/$NAME/preflight/preflight_summary.json" \
    --output "$ROOT/$NAME/formal"
done
