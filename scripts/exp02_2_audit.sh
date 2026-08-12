#!/usr/bin/env bash
set -uo pipefail
ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
PY="$ROOT/.venv/bin/python"
DATA_ROOT="/root/autodl-tmp/borescope-new-seg-data/v1"
WEIGHTS="${WEIGHTS:?WEIGHTS required}"
BATCH="${BATCH:-32}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/evaluation/exp02_2_size_error_audit_$STAMP"
mkdir -p "$OUT"
printf 'weights=%s\ndata_root=%s\nsplit=val\nimgsz=640\nstandard_conf=0.001\nstandard_nms_iou=0.70\ndiagnostic_conf=0.25\ndiagnostic_nms_iou=0.70\nmask_match_iou=0.50\nbatch=%s\n' "$WEIGHTS" "$DATA_ROOT" "$BATCH" > "$OUT/config.txt"
printf '%q ' "$PY" "$ROOT/tools/evaluation/evaluate_seg_model.py" --weights "$WEIGHTS" --data "$DATA_ROOT/data.yaml" --split val --imgsz 640 --conf 0.001 --iou 0.7 --batch "$BATCH" --output "$OUT/standard_val" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
printf '%q ' "$PY" "$ROOT/tools/evaluation/audit_baseline_errors.py" --weights "$WEIGHTS" --data-root "$DATA_ROOT" --manifest "$DATA_ROOT/split_manifest.csv" --output "$OUT/size_error_audit" --batch "$BATCH" >> "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
nvidia-smi > "$OUT/env.txt"
code=0
"$PY" "$ROOT/tools/evaluation/evaluate_seg_model.py" --weights "$WEIGHTS" --data "$DATA_ROOT/data.yaml" --split val --imgsz 640 --conf 0.001 --iou 0.7 --batch "$BATCH" --output "$OUT/standard_val" 2>&1 | tee "$OUT/standard_val.log"
rc="${PIPESTATUS[0]}"; [[ "$rc" -ne 0 ]] && code="$rc"
if [[ "$code" -eq 0 ]]; then
  "$PY" "$ROOT/tools/evaluation/audit_baseline_errors.py" --weights "$WEIGHTS" --data-root "$DATA_ROOT" --manifest "$DATA_ROOT/split_manifest.csv" --output "$OUT/size_error_audit" --batch "$BATCH" 2>&1 | tee "$OUT/size_error_audit.log"
  rc="${PIPESTATUS[0]}"; [[ "$rc" -ne 0 ]] && code="$rc"
fi
sha256sum "$WEIGHTS" > "$OUT/model_sha256.txt"
printf 'return_code=%s\ntest_accessed=false\n' "$code" > "$OUT/summary.txt"
[[ "$code" -eq 0 ]] && printf 'No runtime abnormalities. Fixed diagnostic point only; no threshold sweep.\n' > "$OUT/abnormal.txt" || printf 'Audit failed; see logs.\n' > "$OUT/abnormal.txt"
[[ "$code" -eq 0 ]] && ln -sfn "$(basename "$OUT")" "$ROOT/results/evaluation/exp02_2_latest"
exit "$code"
