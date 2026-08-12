#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/损伤训练数据集}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUT="$PROJECT_ROOT/results/dataset_audit/exp00_4_unpaired_review"
SOURCE="$PROJECT_ROOT/results/dataset_audit/latest/artifacts/near_duplicate_groups.csv"
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
printf '%q ' "$PYTHON_BIN" "$PROJECT_ROOT/tools/dataset/review_unpaired_images.py" --data-root "$DATA_ROOT" --near-groups "$SOURCE" --output-dir "$OUT/artifacts" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
set +e
"$PYTHON_BIN" "$PROJECT_ROOT/tools/dataset/review_unpaired_images.py" --data-root "$DATA_ROOT" --near-groups "$SOURCE" --output-dir "$OUT/artifacts" 2>&1 | tee "$OUT/run.log"
code="${PIPESTATUS[0]}"
set -e
printf 'return_code=%s\n' "$code" > "$OUT/summary.txt"
if [[ "$code" -ne 0 ]]; then printf 'Exp00.4 failed; see run.log\n' > "$OUT/abnormal.txt"; exit "$code"; fi
printf 'No final semantic decisions were made. All 24 rows require human review.\n' > "$OUT/abnormal.txt"
printf 'No model is produced by Exp00.4.\n' > "$OUT/model_sha256.txt"

