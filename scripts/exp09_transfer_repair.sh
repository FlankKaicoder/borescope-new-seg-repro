#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/borescope-new-seg-repro
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec .venv/bin/python tools/evaluation/exp09_transfer_repair.py
