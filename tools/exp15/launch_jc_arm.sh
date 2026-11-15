#!/usr/bin/env bash
# Run one authorized arm. The J1 launcher requires completed independent J0 acceptance.
set -euo pipefail
umask 077
base=$1
arm=$2
case "$arm" in
  j0|j1) ;;
  *) exit 2 ;;
esac
# This file is private and is created during the zero-update preflight.
source "$base/private/runtime.env"
export YOLO_CONFIG_DIR="$base/settings_$arm" YOLO_AUTOINSTALL=false YOLO_OFFLINE=true
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$base/code"
test -s "$base/private/preflight.json"
if [[ "$arm" == j1 ]]; then
  test -f "$base/j0.launch.exit"
  test "$(cut -f2 "$base/j0.launch.exit")" -eq 0
  "$jc_python" -c 'import json,sys; x=json.load(open(sys.argv[1])); sys.exit(0 if x["status"]=="PASS_JC_J0_PRE_J1" and x["arm"]["effective_updates"]==2100 else 1)' "$base/private/j0_accepted.json"
fi
test ! -e "$base/$arm"
test ! -e "$base/formal_budget/$arm.attempts"
test ! -e "$base/$arm.launch.exit"
trap 'code=$?; printf "%s\t%s\n" "$(date -Iseconds)" "$code" > "$base/$arm.launch.exit"' EXIT
"$jc_python" -m tools.exp15.jb formal \
  --release "$jc_release" --weight "$jc_weight" --output "$base/$arm" \
  --budget "$base/formal_budget" --ja-root "$jc_ja" --ja-public "$jc_ja_public" \
  --calibration "$jc_calibration" --protocol "$base/private/formal_protocol.json" \
  --authorize "$base/private/train_authorization.json" --smoke "$jc_smoke" --arm "$arm" &
child=$!
printf "%s\n" "$child" > "$base/$arm.training.pid"
wait "$child"
