#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/borescope-new-seg-repro
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec .venv/bin/python -m tools.training.exp09_simsiam "$@"
