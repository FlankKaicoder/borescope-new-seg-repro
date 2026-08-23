# Exp12.3 downstream A/B handoff

Status: `PASS_SINGLE_SEED_FROZEN_VAL_POSITIVE`.

## Confirmed facts

- Exp12.3-S smoke and Exp12.3 formal Gates both PASS.
- Route B strictly injected only YOLO11 layers 0--10. After Trainer setup it
  matched the Exp12.1 SSL export exactly and differed from official COCO in
  120/120 backbone parameter tensors.
- A/B neck and segmentation head initialization hashes matched exactly before
  training. First-batch stems and augmented tensor hashes also matched.
- Each formal route completed 100/100 epochs, 66,800 TRAIN draws and 1,066
  optimizer steps. TRAIN losses were finite; best/last were finite/reloadable.
- Frozen VAL primary Mask mAP50-95: Route A 0.2989812341, Route B 0.3218971509,
  delta +0.0229159168.
- Mask mAP50 delta +0.0297350745; mask precision delta +0.0683263207; mask
  recall delta -0.0481360253. The result is not uniformly better on every
  metric.
- Route A best SHA256:
  `d053cc65b3b61826d08cfd0f20261b6cdff6050c747ac87cadb66afd64a34c25`.
- Route B best SHA256:
  `da7276bba76955056b5217f241962cf416657abe42d9a105113f1de68b1e477c`.
- TEST was absent from the Exp12.3 YAML and access remained zero.

## Interpretation boundary

This is a positive single-seed, frozen-VAL signal for corrected SimSiam domain
adaptation. It is not statistical confirmation, does not establish TEST
generalization, does not replace the final seed44 Baseline and does not modify
any Exp00--Exp11 conclusion.

The known early C2PSA/AMP validation-loss symptom occurred for epochs 1--5 in
Route A and 1--6 in Route B. TRAIN losses, metric outputs and checkpoint states
were finite; no retry or protocol change was made.

## Evidence

- `docs/exp12_3_downstream_ab.md`
- `results/exp12_downstream_ab/smoke_20260822T085916Z/smoke_gate.json`
- `results/exp12_downstream_ab/formal_20260822T090202Z/formal_gate.json`
- `results/exp12_downstream_ab/formal_20260822T090202Z/frozen_val_comparison.json`
- `tools/training/exp12_3_downstream.py`
- `scripts/exp12_3_downstream_ab.sh`

## Next boundary

Basic SimSiam Exp12.0--12.3 is complete. Multi-scale Local Change Localization
has not been implemented and requires a separate design and authorization Gate.
