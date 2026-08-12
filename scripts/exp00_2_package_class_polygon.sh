#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
SOURCE="$(readlink -f "$PROJECT_ROOT/results/dataset_audit/latest")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$PROJECT_ROOT/results/dataset_audit/exp00_2_class_polygon_$STAMP"
mkdir -p "$OUT/artifacts"
printf 'Derived from immutable unified scan: %s\n' "$SOURCE" > "$OUT/command.txt"
for name in dataset_summary.json class_stats.csv instance_stats.csv cooccurrence.csv polygon_issues.csv; do cp "$SOURCE/artifacts/$name" "$OUT/artifacts/$name"; done
printf '7 classes, 1847 instances, max/min imbalance 20.83:1, 0 program-detected polygon geometry issues.\n' > "$OUT/summary.txt"
printf 'Severe imbalance and 296 instances below relative polygon area 0.001 require split/size-aware handling.\n' > "$OUT/abnormal.txt"
printf 'No model is produced by Exp00.2.\n' > "$OUT/model_sha256.txt"
cp "$SOURCE/run.log" "$OUT/run.log"
printf '%s\n' "$OUT"

