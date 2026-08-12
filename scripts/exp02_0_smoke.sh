#!/usr/bin/env bash
set -uo pipefail
ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"; PY="$ROOT/.venv/bin/python"
MODEL="$ROOT/weights/yolo11n-seg.pt"; DATA="/root/autodl-tmp/borescope-new-seg-data/v1/data.yaml"; BATCH="${BATCH:?BATCH required}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; OUT="$ROOT/results/training/exp02_0_smoke_$STAMP"; mkdir -p "$OUT"
printf '%q ' "$PY" "$ROOT/tools/training/run_yolo_seg.py" --mode smoke --model "$MODEL" --data "$DATA" --output "$OUT/artifacts" --batch "$BATCH" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"; nvidia-smi > "$OUT/env.txt"
"$PY" "$ROOT/tools/training/run_yolo_seg.py" --mode smoke --model "$MODEL" --data "$DATA" --output "$OUT/artifacts" --batch "$BATCH" 2>&1 | tee "$OUT/run.log"; code="${PIPESTATUS[0]}"
printf 'return_code=%s\n' "$code" > "$OUT/summary.txt"; [[ "$code" -eq 0 ]] && printf 'No runtime abnormalities.\n' > "$OUT/abnormal.txt" || printf 'Smoke failed; see run.log.\n' > "$OUT/abnormal.txt"
[[ "$code" -eq 0 ]] && cp "$OUT/artifacts/summary.json" "$OUT/model_sha256.txt" || printf 'No valid smoke checkpoint.\n' > "$OUT/model_sha256.txt"
[[ "$code" -eq 0 ]] && ln -sfn "$(basename "$OUT")" "$ROOT/results/training/exp02_0_smoke_latest"
exit "$code"
