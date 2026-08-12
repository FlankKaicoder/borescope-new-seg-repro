#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$PROJECT_ROOT/results/dataset_audit/exp00_0_environment_$STAMP"
mkdir -p "$OUT/artifacts"
printf 'bash scripts/exp00_environment.sh\n' > "$OUT/command.txt"
cp "$PROJECT_ROOT/environment_report.md" "$OUT/artifacts/environment_report.md"
cp "$PROJECT_ROOT/pip_freeze.txt" "$OUT/artifacts/pip_freeze.txt"
cp "$PROJECT_ROOT/requirements-lock.txt" "$OUT/artifacts/requirements-lock.txt"
printf 'Exp00.0 environment audit completed. Training gate STOP: one visible 22GB RTX 2080 Ti, not two 11GB devices; required training packages are incomplete.\n' > "$OUT/summary.txt"
printf 'Environment differs from task assumption; see summary and environment report.\n' > "$OUT/abnormal.txt"
printf 'No model is produced by Exp00.0.\n' > "$OUT/model_sha256.txt"
cp "$PROJECT_ROOT/environment_report.md" "$OUT/run.log"
printf '%s\n' "$OUT"

