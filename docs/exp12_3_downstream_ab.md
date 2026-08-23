# Exp12.3 downstream A/B comparison

Status: `PASS_SINGLE_SEED_FROZEN_VAL_POSITIVE`.

## Scope and interpretation boundary

Exp12.3 is an independent post-project representation-learning extension. It
does not reopen Exp00--Exp11 model selection, replace the final seed44
Baseline, or alter the frozen TEST conclusion. TEST was not present in the
Exp12.3 data YAML and was not accessed.

The experiment answers a narrow question: after a verified TRAIN-only SimSiam
adaptation, does a strictly injected YOLO11 layers 0--10 initialization change
one controlled seed42 downstream result on the frozen VAL split? A single seed
and one VAL split do not establish a statistically robust or TEST-generalized
improvement.

## Frozen protocol

- Route A: official COCO `yolo11n-seg.pt` -> supervised fine-tuning.
- Route B: the same model construction -> strict Exp12.1 layers 0--10 injection
  -> supervised fine-tuning.
- Both routes: frozen TRAIN/VAL split, image size 640, batch 32, 100 epochs,
  AdamW, seed 42, deterministic mode, AMP enabled, workers 4 and patience 100.
- Route B injection occurs only after the 7-class segmentation model is built.
  This preserves the random initialization sequence for neck and Segment head.
- Selection: each route's Ultralytics `best.pt` under the same frozen VAL
  fitness. Final reporting uses an additional identical frozen-VAL evaluation.

## Exp12.3-S smoke Gate

Run: `results/exp12_downstream_ab/smoke_20260822T085916Z`.

- Both routes completed one epoch, 668/668 TRAIN image draws and finite TRAIN
  losses; best/last checkpoints were finite and reloadable.
- Route B matched the SSL export after Trainer setup and differed from official
  COCO in 120/120 backbone parameter tensors (`changed_ratio=1.0`).
- Route A and Route B neck/head hashes were identical both immediately after
  model construction and after Trainer setup.
- First-batch stems and the augmented input hash were identical:
  `2ee95ef7f78d42707bc660966b073c636299ce70970d7dad4a42da03f2497b2f`.
- All Gate checks passed and TEST access was zero.

## Formal execution audit

Run: `results/exp12_downstream_ab/formal_20260822T090202Z`.

| Check | Route A | Route B |
|---|---:|---:|
| Epochs | 100/100 | 100/100 |
| TRAIN image draws | 66,800/66,800 | 66,800/66,800 |
| Optimizer steps | 1,066 | 1,066 |
| TRAIN losses finite | PASS | PASS |
| best/last finite and reloadable | PASS | PASS |
| TEST access | 0 | 0 |

The known C2PSA/AMP validation-loss accounting symptom appeared at epochs 1--5
for Route A and epochs 1--6 for Route B. TRAIN losses, checkpoint states and
validation metrics remained finite; no retry, recovery, hyperparameter change
or early termination was performed. This is retained as an audit caveat.

## Frozen VAL result

| Metric | Route A | Route B | B - A |
|---|---:|---:|---:|
| Mask mAP50-95 (primary) | 0.298981 | 0.321897 | +0.022916 |
| Mask mAP50 | 0.531920 | 0.561655 | +0.029735 |
| Mask precision | 0.662923 | 0.731249 | +0.068326 |
| Mask recall | 0.523024 | 0.474888 | -0.048136 |
| Box mAP50-95 | 0.337830 | 0.354354 | +0.016524 |
| Box mAP50 | 0.593754 | 0.587395 | -0.006359 |
| Fitness | 0.636812 | 0.676251 | +0.039440 |

Route A best checkpoint SHA256 is
`d053cc65b3b61826d08cfd0f20261b6cdff6050c747ac87cadb66afd64a34c25`;
Route B best checkpoint SHA256 is
`da7276bba76955056b5217f241962cf416657abe42d9a105113f1de68b1e477c`.

## Conclusion

The engineering claim is confirmed: the corrected SimSiam path really changes
the YOLO backbone and survives strict downstream Trainer construction. Under
