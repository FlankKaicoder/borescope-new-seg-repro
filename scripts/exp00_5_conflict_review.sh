#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/损伤训练数据集}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUT="$PROJECT_ROOT/results/dataset_audit/exp00_5_conflict_review"
SOURCE="$PROJECT_ROOT/results/dataset_audit/latest/artifacts"
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
printf '%q ' "$PYTHON_BIN" "$PROJECT_ROOT/tools/dataset/review_conflict_groups.py" --data-root "$DATA_ROOT" --groups-csv "$SOURCE/near_duplicate_groups.csv" --pairs-csv "$SOURCE/near_duplicate_pairs.csv" --consistency-csv "$SOURCE/near_duplicate_annotation_consistency.csv" --output-dir "$OUT/artifacts" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
set +e
"$PYTHON_BIN" "$PROJECT_ROOT/tools/dataset/review_conflict_groups.py" --data-root "$DATA_ROOT" --groups-csv "$SOURCE/near_duplicate_groups.csv" --pairs-csv "$SOURCE/near_duplicate_pairs.csv" --consistency-csv "$SOURCE/near_duplicate_annotation_consistency.csv" --output-dir "$OUT/artifacts" 2>&1 | tee "$OUT/run.log"
code="${PIPESTATUS[0]}"
set -e
printf 'return_code=%s\n' "$code" > "$OUT/summary.txt"
if [[ "$code" -ne 0 ]]; then printf 'Exp00.5 failed; see run.log\n' > "$OUT/abnormal.txt"; exit "$code"; fi
printf 'No image was deleted and no JSON or label was changed. All 22 groups require human review.\n' > "$OUT/abnormal.txt"
printf 'No model is produced by Exp00.5.\n' > "$OUT/model_sha256.txt"

