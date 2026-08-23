# Exp12.5 final review handoff

## Status

`EXP12_COMPLETE_LOCAL_CHANGE_VALID_PROXY_NO_FROZEN_VAL_MASK_GAIN`

## Verified facts

- Exp12.4-S and 30-epoch formal TRAIN-only local-change SSL both PASS.
- Backbone first-step and final update audits: 120/120 parameter tensors; formal
  final changed all 1,365,472 elements.
- Formal checkpoint and backbone export reload match 120/120 parameter hashes.
- Exp12.5 Route C injection, Trainer initialization, neck/head, first-batch and
  training-argument fairness Gates all PASS.
- Route C completed 100 epochs, 66,800 TRAIN draws and 1,066 optimizer steps.
- Frozen VAL Mask mAP50-95: A 0.298981, B 0.321897, C 0.293530.
- Route C delta: -0.005451 vs A and -0.028367 vs B.
- TEST access is zero for Exp12.4 and Exp12.5.

## Interpretation

The proxy method is implemented correctly and changes the intended backbone,
but its single-seed frozen-VAL segmentation result is negative. Proxy IoU
0.952284 must not be presented as downstream benefit. Exp00--Exp11 conclusions
remain unchanged; Exp12 does not justify TEST access or another search round.

## Canonical artifacts

- `docs/exp12_4_5_local_change.md`
- `results/exp12_local_change/smoke_20260823T064652Z/summary.json`
- `results/exp12_local_change/formal_20260823T064923Z/summary.json`
- `results/exp12_local_change/formal_20260823T064923Z/parameter_update_report.json`
- `results/exp12_local_change_downstream/smoke_20260823T070449Z/smoke_gate.json`
- `results/exp12_local_change_downstream/formal_20260823T070600Z/formal_gate.json`
- `results/exp12_local_change_downstream/formal_20260823T070600Z/frozen_val_comparison.json`

The earlier smoke `smoke_20260823T070045Z` is intentionally retained as the
path-normalization-only `HARD_GATE` attempt.
