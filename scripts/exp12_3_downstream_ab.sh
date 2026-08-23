#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/borescope-new-seg-repro
PYTHON="$REPO/.venv/bin/python"
SCRIPT="$REPO/tools/training/exp12_3_downstream.py"
PHASE="${1:?usage: exp12_3_downstream_ab.sh smoke|formal OUTPUT [SMOKE_GATE]}"
OUTPUT="${2:?output directory is required}"

if [[ "$PHASE" == "smoke" ]]; then
  exec "$PYTHON" "$SCRIPT" pair --phase smoke --output "$OUTPUT"
fi

if [[ "$PHASE" == "formal" ]]; then
  SMOKE_GATE="${3:?formal phase requires the PASS smoke_gate.json path}"
  exec "$PYTHON" "$SCRIPT" pair --phase formal --output "$OUTPUT" --smoke-gate "$SMOKE_GATE"
fi

echo "Unsupported phase: $PHASE" >&2
exit 2
