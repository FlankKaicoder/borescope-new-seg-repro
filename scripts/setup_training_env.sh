#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
BASE_PYTHON="${BASE_PYTHON:-/root/miniconda3/bin/python}"
VENV="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/results/environment/exp00_training_env"
mkdir -p "$LOG_DIR"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel 2>&1 | tee "$LOG_DIR/install_bootstrap.log"
"$VENV/bin/python" -m pip install \
  opencv-python-headless \
  pandas \
  scikit-learn \
  shapely \
  scipy \
  py-cpuinfo \
  polars \
  nvidia-ml-py \
  ultralytics-thop \
  2>&1 | tee "$LOG_DIR/install_dependencies.log"
# Ultralytics declares opencv-python, while this headless server deliberately uses
# opencv-python-headless. Install Ultralytics itself without pulling the GUI wheel.
"$VENV/bin/python" -m pip install --no-deps ultralytics 2>&1 | tee "$LOG_DIR/install_ultralytics.log"

"$VENV/bin/python" "$PROJECT_ROOT/tools/environment/smoke_training_env.py" \
  --image '/root/autodl-tmp/损伤训练数据集/1.png' \
  --output "$LOG_DIR/smoke_report.json" \
  2>&1 | tee "$LOG_DIR/smoke.log"

"$VENV/bin/python" -m pip freeze > "$PROJECT_ROOT/training_pip_freeze.txt"
"$VENV/bin/python" -m pip freeze > "$PROJECT_ROOT/training_requirements-lock.txt"
"$VENV/bin/python" "$PROJECT_ROOT/tools/environment/report_training_env.py" \
  --smoke "$LOG_DIR/smoke_report.json" \
  --output "$PROJECT_ROOT/training_environment_report.md"
