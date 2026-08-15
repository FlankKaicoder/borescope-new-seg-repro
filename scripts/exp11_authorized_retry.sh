#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/borescope-new-seg-repro

WEIGHTS=/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt
DATA_ROOT=/root/autodl-tmp/borescope-new-seg-data/v1
MANIFEST=/root/autodl-tmp/borescope-new-seg-data/v1/split_manifest.csv
FREEZE=/root/autodl-tmp/borescope-new-seg-repro/results/final_test/candidate_freeze.json
ORIGINAL=/root/autodl-tmp/borescope-new-seg-repro/results/final_test/exp11
AUTH=/root/autodl-tmp/borescope-new-seg-repro/results/final_test/exp11_retry_authorization.json
OUTPUT=/root/autodl-tmp/borescope-new-seg-repro/results/final_test/exp11_retry1

test -f "$AUTH"
test -f "$ORIGINAL/candidate_freeze.json"
test -f "$ORIGINAL/checkpoint_sha256.txt"
test -f "$ORIGINAL/command.txt"
test -f "$ORIGINAL/environment.json"
test ! -e "$OUTPUT"

.venv/bin/python tools/evaluation/exp11_final_test.py \
  --weights "$WEIGHTS" \
  --data "$DATA_ROOT/data.yaml" \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --freeze "$FREEZE" \
  --output "$OUTPUT"
