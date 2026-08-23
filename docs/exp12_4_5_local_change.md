# Exp12.4--12.5 multi-scale local-change extension

## Scope and data boundary

This work was explicitly authorized after Exp12.3 and remains an independent
post-project research extension. It does not replace the final Exp00--Exp11
Baseline or alter any TEST conclusion.

Exp12.4 SSL used only the exact frozen 668-image `images/train` set. The input
Gate verified every image symlink against the frozen read-only source and the
split manifest. SSL VAL/TEST access was 0/0. Exp12.5 used the frozen supervised
TRAIN/VAL YAML from Exp12.3; TEST was absent and never accessed.

## Exp12.4 method

The new implementation creates a spatially aligned pair `(x, x')` from each
TRAIN image with one to three synthetic local changes: same-image texture
transplant, localized photometric/contrast change, or localized blur. It emits
the exact binary change mask. This is a new implementation for the new
borescope dataset; historical gas-pore code was not copied.

YOLO11n-seg layers 0--10 form the shared encoder. Absolute paired differences
from layers 4, 6 and 10 (P3/P4/P5, channels 128/128/256) are projected, aligned
to P3, concatenated and decoded by a small mask head. Training uses focal BCE
plus soft Dice loss. All 120 backbone parameter tensors are explicitly
unfrozen and audited before/after training.

## Exp12.4-S smoke

Run: `results/exp12_local_change/smoke_20260823T064652Z`.

- TRAIN exact-set Gate: 668 images; 41 steps and 656 samples with batch 16 and
  `drop_last=True`.
- First real batch: gradients present/finite/non-zero for 120/120 backbone
  tensors; 1,365,441/1,365,472 gradient elements were non-zero.
- First optimizer step changed 120/120 tensors and all 1,365,472 elements.
- One full epoch changed 120/120 tensors and all elements; `changed_ratio=1.0`.
- Loss 0.605465; proxy IoU 0.565421; feature std 0.228425; logit std 1.080904.
- Checkpoint and backbone export each reloaded with 120/120 parameter hashes.
- Status: `PASS`; `INVALID_BY_BACKBONE_NO_UPDATE` was not triggered.

## Exp12.4 formal SSL and audit

Run: `results/exp12_local_change/formal_20260823T064923Z`.

- Fixed 30/30 epochs, 41 steps and 656 samples per epoch; no early stopping,
  VAL selection or TEST access.
- Loss moved from 0.601618 to 0.030624; proxy IoU from 0.573189 to 0.952284.
- Minimum feature std 0.228837 and minimum logit std 1.077156; no collapse Gate
  breach, NaN, Inf or OOM.
- Final audit: 120/120 backbone parameter tensors and all 1,365,472 elements
  changed; all parameters finite.
- Final checkpoint SHA256:
  `7e789c11ae5551a9fce68c95f9992bf0e217d43b6e1b2ff660892523db5e9f48`.
- Backbone export SHA256:
  `e0358d2674752fb7a151a4fe9f6ef1a1d58bc0cf0a60de3b55ad61b5060bdbf0`.
- Checkpoint/export round trip: 120/120 hashes for both; status `PASS`.

The proxy task therefore trained the YOLO backbone correctly. High proxy IoU
is only an implementation/task-learning diagnostic and is not evidence of
downstream benefit.

## Exp12.5 downstream fairness Gates

Route definitions:

- Route A: locked Exp12.3 official COCO initialization.
- Route B: locked Exp12.3 SimSiam initialization.
- Route C: Exp12.4 fixed-final local-change backbone initialization.

Only Route C was newly trained. Reusing locked A/B avoids redundant compute and
preserves their exact checkpoints. Route C used the same frozen YAML, 100
epochs, batch 32, AdamW, seed 42, augmentation and validation policy. Neck/head
hashes, first TRAIN sample stems and first input tensor hash matched A/B.
Route C differed from official COCO in 120/120 backbone parameters and remained
exactly equal to the Exp12.4 export after Trainer setup.

The first smoke run at
`results/exp12_local_change_downstream/smoke_20260823T070045Z` is retained as a
`HARD_GATE`: all model/training checks passed, but fairness comparison treated a
relative and absolute path to the same YAML as different strings. No protocol
or model parameter changed. The v2 launcher normalized the resolved path and
file hash; the new run
`results/exp12_local_change_downstream/smoke_20260823T070449Z` passed every
fairness Gate.

Formal Route C run:
`results/exp12_local_change_downstream/formal_20260823T070600Z`.

- Completed 100/100 epochs, 66,800 TRAIN draws and 1,066 optimizer steps.
- TRAIN losses were finite; best/last checkpoints passed torch and YOLO reload.
- Best checkpoint SHA256:
  `d03fbf1574d5c8ebc40125556a0b854e0268e3dc54d0f4fd5e6a7d3ae68d66db`.
- No retry or hyperparameter modification; TEST access remained zero.

## Frozen VAL comparison (seed 42)

| Route | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | Box mAP50-95 |
|---|---:|---:|---:|---:|---:|
| A: COCO | 0.662923 | 0.523024 | 0.531920 | 0.298981 | 0.337830 |
| B: SimSiam | 0.731249 | 0.474888 | 0.561655 | 0.321897 | 0.354354 |
| C: Local change | 0.669685 | 0.486716 | 0.530281 | 0.293530 | 0.340935 |

Primary Mask mAP50-95 deltas:

- Route C minus A: `-0.005451`.
- Route C minus B: `-0.028367`.

Route C increased Box precision by 0.031198 and Box mAP50-95 by 0.003104 over
Route A, but reduced Box recall by 0.066504, Mask recall by 0.036307 and the
primary Mask mAP50-95 by 0.005451. It therefore does not show a positive
segmentation-representation signal under this single-seed frozen-VAL protocol.

## Conclusion boundary

Exp12.4 proves that a multi-scale local-change proxy task can genuinely update
the YOLO11 backbone and learn its synthetic localization task without collapse.
Exp12.5 shows that this successful proxy optimization did not translate into a
Mask mAP50-95 gain on the frozen VAL split. Ordinary SimSiam Route B remains the
best of the three Exp12 routes on this single seed.

This is not a TEST generalization claim, not multi-seed evidence, and not a
change to the final Baseline. Exp12 is complete; no further model selection or
TEST evaluation is authorized by these results.
