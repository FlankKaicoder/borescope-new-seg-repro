#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
SOURCE="$(readlink -f "$PROJECT_ROOT/results/dataset_audit/latest")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$PROJECT_ROOT/results/dataset_audit/exp00_3_duplicate_$STAMP"
mkdir -p "$OUT/artifacts"
printf 'Derived from immutable unified scan: %s\n' "$SOURCE" > "$OUT/command.txt"
for name in dataset_summary.json duplicate_groups.csv near_duplicate_groups.csv near_duplicate_pairs.csv near_duplicate_annotation_consistency.csv image_hashes.csv; do cp "$SOURCE/artifacts/$name" "$OUT/artifacts/$name"; done
cp -r "$SOURCE/artifacts/near_duplicate_contact_sheets" "$OUT/artifacts/near_duplicate_contact_sheets"
printf '0 exact groups; 88 near-duplicate groups covering 252 images; 37 groups have annotation-signature conflicts, including 22 label-set conflicts.\n' > "$OUT/summary.txt"
printf 'Leakage and annotation consistency gate STOP until groups and conflicting labels are reviewed.\n' > "$OUT/abnormal.txt"
printf 'No model is produced by Exp00.3.\n' > "$OUT/model_sha256.txt"
cp "$SOURCE/run.log" "$OUT/run.log"
printf '%s\n' "$OUT"

