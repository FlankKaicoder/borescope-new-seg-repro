#!/usr/bin/env bash
set -euo pipefail

# AutoDL no-GPU export only.  The source tree is read; the destination must be
# a new directory and is never the immutable raw-data root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-/root/autodl-tmp/损伤训练数据集}"
MANIFEST="${MANIFEST:-/root/autodl-tmp/borescope-new-seg-data/v1/split_manifest.csv}"
OUTPUT="${OUTPUT:?Set OUTPUT to a new TRAIN export directory}"
CLASS_MAPPING="${CLASS_MAPPING:-}"

args=("$ROOT/tools/exp13_cloud/g3/verify_train_export.py" --mode export
  --manifest "$MANIFEST" --source-root "$SOURCE_ROOT" --output "$OUTPUT")
if [[ -n "$CLASS_MAPPING" ]]; then
  args+=(--class-mapping "$CLASS_MAPPING")
fi
exec "$PYTHON" "${args[@]}"
