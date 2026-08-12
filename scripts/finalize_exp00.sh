#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT='/root/autodl-tmp/borescope-new-seg-repro'
cd "$PROJECT_ROOT"
if [[ "$PWD" != "$PROJECT_ROOT" ]]; then
  echo 'Unexpected project path; refusing cleanup.' >&2
  exit 2
fi

# Remove only the duplicate registry accidentally created by the final sync.
if [[ -f "$PROJECT_ROOT/experiment_registry.csv" ]]; then
  rm -f -- "$PROJECT_ROOT/experiment_registry.csv"
fi

sed -i 's/\r$//' results/experiment_registry.csv
git add results/experiment_registry.csv scripts/finalize_exp00.sh
git -c user.name=Codex -c user.email=codex@local commit -m 'fix: store experiment registry in canonical results path'

echo '__GIT__'
git status --short --branch
git log -1 --oneline
echo '__REGISTRY__'
wc -l results/experiment_registry.csv
echo '__FINAL_ARTIFACTS__'
find results/dataset_audit/exp00_[0-3]_* -maxdepth 2 -type f | sort
echo '__SOURCE_READONLY_CHECK__'
find '/root/autodl-tmp/损伤训练数据集' -type f | wc -l
cat results/dataset_audit/latest/artifacts/raw_file_manifest_sha256.txt
