#!/usr/bin/env bash
set -u

echo '__PATH__'
printf '%s\n' "$PATH"

echo '__PYTHON_CANDIDATES__'
for candidate in \
  /root/miniconda3/bin/python \
  /root/miniconda/bin/python \
  /opt/conda/bin/python \
  /usr/bin/python3 \
  /usr/local/bin/python3
do
  if [[ -x "$candidate" ]]; then
    "$candidate" -V 2>&1
    printf '%s\n' "$candidate"
  fi
done

echo '__ROOT__'
ls -la /root | head -n 40

echo '__TOOLS__'
gcc --version 2>&1 | head -n 2 || true
git --version 2>&1 || true

echo '__STORAGE__'
df -h

echo '__MEMORY__'
free -h

echo '__LABEL__'
sed -n '1,100p' '/root/autodl-tmp/损伤训练数据集/label.txt'

echo '__JSON_SAMPLE__'
sample_json="$(find '/root/autodl-tmp/损伤训练数据集' -maxdepth 1 -type f -iname '*.json' | sort -V | head -n 1)"
if [[ -n "$sample_json" ]]; then
  sed -n '1,120p' "$sample_json"
fi
