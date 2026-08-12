#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/损伤训练数据集}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$PROJECT_ROOT/results/dataset_audit/exp00_audit_$STAMP"

mkdir -p "$RUN_DIR/artifacts"
printf '%q ' "$PYTHON_BIN" "$PROJECT_ROOT/tools/dataset/audit_dataset.py" --data-root "$DATA_ROOT" --output-dir "$RUN_DIR/artifacts" --report "$PROJECT_ROOT/docs/01_dataset_audit.md" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

set +e
"$PYTHON_BIN" "$PROJECT_ROOT/tools/dataset/audit_dataset.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$RUN_DIR/artifacts" \
  --report "$PROJECT_ROOT/docs/01_dataset_audit.md" \
  2>&1 | tee "$RUN_DIR/run.log"
return_code="${PIPESTATUS[0]}"
set -e

printf 'return_code=%s\n' "$return_code" > "$RUN_DIR/summary.txt"
if [[ "$return_code" -ne 0 ]]; then
  printf 'Exp00 audit failed; see run.log\n' > "$RUN_DIR/abnormal.txt"
  exit "$return_code"
fi
"$PYTHON_BIN" "$PROJECT_ROOT/tools/visualization/render_near_duplicate_groups.py" \
  --data-root "$DATA_ROOT" \
  --groups-csv "$RUN_DIR/artifacts/near_duplicate_groups.csv" \
  --output-dir "$RUN_DIR/artifacts/near_duplicate_contact_sheets" \
  2>&1 | tee -a "$RUN_DIR/run.log"
printf 'No runtime abnormality. Data issues are recorded in artifacts/polygon_issues.csv and docs/01_dataset_audit.md.\n' > "$RUN_DIR/abnormal.txt"
ln -sfn "$(basename "$RUN_DIR")" "$PROJECT_ROOT/results/dataset_audit/latest"
printf '%s\n' "$RUN_DIR"
