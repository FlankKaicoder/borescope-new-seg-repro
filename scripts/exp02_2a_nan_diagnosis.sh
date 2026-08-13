#!/usr/bin/env bash
set -uo pipefail
ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
PY="$ROOT/.venv/bin/python"
DATA="/root/autodl-tmp/borescope-new-seg-data/v1/data.yaml"
MODEL="$ROOT/weights/yolo11n-seg.pt"
ORIGINAL="$ROOT/results/training/exp02_1_baseline_20260812T135254Z/artifacts/ultralytics/baseline/results.csv"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/diagnostics/exp02_2a_early_val_nan_$STAMP"
mkdir -p "$OUT"
printf 'Exp02.2a only; test forbidden; no full 100-epoch rerun.\n' > "$OUT/config.txt"
printf '%q ' "$PY" "$ROOT/tools/evaluation/exp02_2a_source_audit.py" --output "$OUT/source_audit" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
printf '%q ' "$PY" "$ROOT/tools/evaluation/exp02_2a_early_epoch_audit.py" --results-csv "$ORIGINAL" --output "$OUT/original_early_epoch_audit" >> "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
printf '%q ' "$PY" "$ROOT/tools/training/exp02_2a_short_repro.py" --model "$MODEL" --data "$DATA" --output "$OUT/short_repro" >> "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
printf '%q ' "$PY" "$ROOT/tools/evaluation/exp02_2a_val_loss_probe.py" --manifest "$OUT/short_repro/summary.json" --data "$DATA" --output "$OUT/val_loss_probe" >> "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
printf '%q ' "$PY" "$ROOT/tools/evaluation/exp02_2a_forward_trace.py" --checkpoint "$OUT/short_repro/raw_ema_before_validation/epoch1_raw_ema.pt" --data "$DATA" --output "$OUT/forward_trace" >> "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
printf '%q ' "$PY" "$ROOT/tools/evaluation/exp02_2a_finalize.py" --run "$OUT" --baseline-best "$ROOT/results/training/exp02_1_baseline_20260812T135254Z/artifacts/ultralytics/baseline/weights/best.pt" >> "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
nvidia-smi > "$OUT/env.txt"; "$PY" -m pip freeze > "$OUT/pip_freeze.txt"
code=0
"$PY" "$ROOT/tools/evaluation/exp02_2a_source_audit.py" --output "$OUT/source_audit" 2>&1 | tee "$OUT/source_audit.log"; rc="${PIPESTATUS[0]}"; [[ "$rc" -ne 0 ]] && code="$rc"
if [[ "$code" -eq 0 ]]; then "$PY" "$ROOT/tools/evaluation/exp02_2a_early_epoch_audit.py" --results-csv "$ORIGINAL" --output "$OUT/original_early_epoch_audit" 2>&1 | tee "$OUT/original_audit.log"; rc="${PIPESTATUS[0]}"; [[ "$rc" -ne 0 ]] && code="$rc"; fi
if [[ "$code" -eq 0 ]]; then "$PY" "$ROOT/tools/training/exp02_2a_short_repro.py" --model "$MODEL" --data "$DATA" --output "$OUT/short_repro" 2>&1 | tee "$OUT/short_repro.log"; rc="${PIPESTATUS[0]}"; [[ "$rc" -ne 0 ]] && code="$rc"; fi
if [[ "$code" -eq 0 ]]; then "$PY" "$ROOT/tools/evaluation/exp02_2a_val_loss_probe.py" --manifest "$OUT/short_repro/summary.json" --data "$DATA" --output "$OUT/val_loss_probe" 2>&1 | tee "$OUT/val_loss_probe.log"; rc="${PIPESTATUS[0]}"; [[ "$rc" -ne 0 ]] && code="$rc"; fi
if [[ "$code" -eq 0 ]]; then "$PY" "$ROOT/tools/evaluation/exp02_2a_forward_trace.py" --checkpoint "$OUT/short_repro/raw_ema_before_validation/epoch1_raw_ema.pt" --data "$DATA" --output "$OUT/forward_trace" 2>&1 | tee "$OUT/forward_trace.log"; rc="${PIPESTATUS[0]}"; [[ "$rc" -ne 0 ]] && code="$rc"; fi
if [[ "$code" -eq 0 ]]; then "$PY" "$ROOT/tools/evaluation/exp02_2a_finalize.py" --run "$OUT" --baseline-best "$ROOT/results/training/exp02_1_baseline_20260812T135254Z/artifacts/ultralytics/baseline/weights/best.pt" 2>&1 | tee "$OUT/finalize.log"; rc="${PIPESTATUS[0]}"; [[ "$rc" -ne 0 ]] && code="$rc"; fi
printf 'return_code=%s\ntest_accessed=false\n' "$code" > "$OUT/summary.txt"
[[ "$code" -eq 0 ]] && printf 'No execution abnormality; scientific NaN findings are recorded in CSV/JSON.\n' > "$OUT/abnormal.txt" || printf 'Execution failure; see logs.\n' > "$OUT/abnormal.txt"
[[ "$code" -eq 0 ]] && ln -sfn "$(basename "$OUT")" "$ROOT/results/diagnostics/exp02_2a_latest"
exit "$code"
