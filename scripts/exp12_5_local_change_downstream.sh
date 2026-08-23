#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 smoke|formal [passed_smoke_gate.json]" >&2
  exit 2
fi

PHASE="$1"
SMOKE_GATE="${2:-}"
ROOT=/root/autodl-tmp/borescope-new-seg-repro
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/exp12_local_change_downstream/${PHASE}_${RUN_ID}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$PHASE" != smoke && "$PHASE" != formal ]]; then
  echo "phase must be smoke or formal" >&2
  exit 2
fi
if [[ "$PHASE" == formal && ! -f "$SMOKE_GATE" ]]; then
  echo "formal phase requires a passed smoke gate JSON" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing Exp12.5 output: $OUT" >&2
  exit 2
fi

ARGS=(
  --phase "$PHASE"
  --output "$OUT"
  --local-backbone "$ROOT/results/exp12_local_change/formal_20260823T064923Z/adapted_backbone_final.pt"
  --local-formal-summary "$ROOT/results/exp12_local_change/formal_20260823T064923Z/summary.json"
)
if [[ "$PHASE" == formal ]]; then
  ARGS+=(--smoke-gate "$SMOKE_GATE")
fi

exec .venv/bin/python -m tools.training.exp12_5_local_change_downstream "${ARGS[@]}"
