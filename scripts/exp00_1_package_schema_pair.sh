#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
SOURCE="$(readlink -f "$PROJECT_ROOT/results/dataset_audit/latest")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$PROJECT_ROOT/results/dataset_audit/exp00_1_schema_pair_$STAMP"
mkdir -p "$OUT/artifacts"
printf 'Derived from immutable unified scan: %s\n' "$SOURCE" > "$OUT/command.txt"
for name in dataset_summary.json schema_samples.csv missing_pairs.csv decode_failed_images.csv json_parse_failed.csv raw_file_manifest.csv raw_file_manifest_sha256.txt image_without_json_contact_sheet.jpg source_field_audit.json; do cp "$SOURCE/artifacts/$name" "$OUT/artifacts/$name"; done
printf '993 images, 969 JSON, 969 matched, 24 images without JSON, 0 JSON without image, 0 decode/parse failures.\n' > "$OUT/summary.txt"
printf '24 readable images have no JSON; their intended background/unlabeled semantics require confirmation.\n' > "$OUT/abnormal.txt"
printf 'No model is produced by Exp00.1.\n' > "$OUT/model_sha256.txt"
cp "$SOURCE/run.log" "$OUT/run.log"
printf '%s\n' "$OUT"

