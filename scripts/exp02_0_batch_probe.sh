#!/usr/bin/env bash
set -uo pipefail
ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"; PY="$ROOT/.venv/bin/python"
MODEL="$ROOT/weights/yolo11n-seg.pt"; DATA="/root/autodl-tmp/borescope-new-seg-data/v1/data.yaml"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; OUT="$ROOT/results/training/exp02_0_batch_probe_$STAMP"; mkdir -p "$OUT"
printf 'batches=8,16,24,32;imgsz=640;fraction=0.03;real_train_steps=true\n' > "$OUT/command.txt"; nvidia-smi > "$OUT/env.txt"
code=0
for batch in 8 16 24 32; do
  "$PY" "$ROOT/tools/training/run_yolo_seg.py" --mode probe --model "$MODEL" --data "$DATA" --output "$OUT/batch_$batch" --batch "$batch" --fraction 0.03 2>&1 | tee "$OUT/batch_$batch.log"
  rc="${PIPESTATUS[0]}"; printf 'batch=%s return_code=%s\n' "$batch" "$rc" >> "$OUT/probe_status.txt"
  [[ "$rc" -ne 0 && "$rc" -ne 42 ]] && code="$rc"
done
printf 'return_code=%s\n' "$code" > "$OUT/summary.txt"; printf 'See per-batch summary.json and logs.\n' > "$OUT/abnormal.txt"; printf 'No retained model is produced by batch probe.\n' > "$OUT/model_sha256.txt"
[[ "$code" -eq 0 ]] && ln -sfn "$(basename "$OUT")" "$ROOT/results/training/exp02_0_probe_latest"
exit "$code"
