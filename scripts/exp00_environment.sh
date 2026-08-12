#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/borescope-new-seg-repro}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUT="$PROJECT_ROOT/environment_report.md"

{
  echo '# Exp00.0 Environment Audit'
  echo
  echo "Generated UTC: $(date -u --iso-8601=seconds)"
  echo
  for title in nvidia-smi uname os-release python gcc df free; do
    echo "## $title"
    echo
    echo '```text'
    case "$title" in
      nvidia-smi) nvidia-smi 2>&1 ;;
      uname) uname -a 2>&1 ;;
      os-release) cat /etc/os-release 2>&1 ;;
      python) "$PYTHON_BIN" -V 2>&1 ;;
      gcc) gcc --version 2>&1 ;;
      df) df -h 2>&1 ;;
      free) free -h 2>&1 ;;
    esac
    echo '```'
    echo
  done
  echo '## Python packages'
  echo
  echo '| Import | Version/status |'
  echo '|---|---|'
  "$PYTHON_BIN" - <<'PY'
import importlib
mods=['torch','torchvision','ultralytics','cv2','numpy','sklearn','pandas','matplotlib','PIL','shapely']
for mod in mods:
    try:
        package=importlib.import_module(mod)
        value=getattr(package,'__version__','installed')
    except Exception as exc:
        value=f'MISSING ({type(exc).__name__}: {exc})'
    print(f'| `{mod}` | {value} |')
PY
  echo
  echo '## CUDA visibility check'
  echo
  echo '```text'
  "$PYTHON_BIN" - <<'PY'
import torch
print('torch.version.cuda=', torch.version.cuda)
print('torch.cuda.is_available=', torch.cuda.is_available())
print('torch.cuda.device_count=', torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    props=torch.cuda.get_device_properties(index)
    print(index, props.name, props.total_memory, props.total_memory/1024**2, 'MiB')
PY
  echo '```'
} > "$OUT"

"$PYTHON_BIN" -m pip freeze > "$PROJECT_ROOT/pip_freeze.txt"

