#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/borescope-new-seg-data/v1}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; OUT="$PROJECT_ROOT/results/dataset_build/exp01_2_verification_$STAMP"
mkdir -p "$OUT"
printf '%q ' "$PYTHON_BIN" "$PROJECT_ROOT/tools/dataset/build_dataset_v1.py" verify --dataset-root "$DATASET_ROOT" --output "$OUT/artifacts" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/dataset/build_dataset_v1.py" verify --dataset-root "$DATASET_ROOT" --output "$OUT/artifacts" 2>&1 | tee "$OUT/run.log"
code="${PIPESTATUS[0]}"; printf 'return_code=%s\n' "$code" > "$OUT/summary.txt"
[[ "$code" -eq 0 ]] && printf 'No runtime abnormalities. Seven-class overlay coverage and Dataset Freeze verification PASS.\n' > "$OUT/abnormal.txt" || printf 'Verification failed; see run.log. Dataset Freeze Gate STOP.\n' > "$OUT/abnormal.txt"
printf 'No model is produced by Exp01.2 verification.\n' > "$OUT/model_sha256.txt"
[[ "$code" -eq 0 ]] && ln -sfn "$(basename "$OUT")" "$PROJECT_ROOT/results/dataset_build/exp01_2_verification_latest"
exit "$code"
