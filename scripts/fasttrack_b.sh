#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/borescope-new-seg-repro
export PYTHONPATH="$PWD/tools/evaluation:${PYTHONPATH:-}"
exec .venv/bin/python tools/fasttrack_b.py "$@"
