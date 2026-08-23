# Exp12.4 multi-scale local-change SSL plan

## Scope

Exp12.4 is an independently authorized extension after Exp12.0--12.3. It does
not change the closed Exp00--Exp11 conclusions. SSL input is exactly the 668
frozen `images/train` symlinks; VAL and TEST access are forbidden.

## Encoder and proxy task

- Start from official `yolo11n-seg.pt` and deep-copy backbone layers 0--10.
- Explicitly unfreeze all 120 parameter tensors (1,365,472 elements).
- Tap layers 4, 6 and 10 (P3/P4/P5); runtime channel counts are audited.
- Build a spatially aligned pair `(x, x')` by adding one to three local changes
  to `x`: same-image texture transplant, localized photometric change, or
  localized blur. The generator also emits the exact binary change mask.
- Compute absolute paired feature differences at all three scales, project each
  to 64 channels, align to P3, concatenate, and predict the mask with a small
  change head.
- Optimize focal BCE plus soft Dice loss. The proxy task is newly implemented
  for the new borescope dataset; historical gas-pore code is not copied.

## Gate sequence

1. TRAIN exact-set/symlink/read-only-source Gate.
2. Structure, official-weight hash alignment and explicit-unfreeze Gate.
3. Optimizer object-identity Gate.
4. First real batch gradient and optimizer-step Gate.
5. One-epoch smoke: finite metrics, non-empty masks, no sustained feature/head
   variance collapse, parameter/checkpoint/export round trip.
6. If `changed_ratio == 0`, emit `INVALID_BY_BACKBONE_NO_UPDATE` and stop.
7. Only after smoke PASS, run a fixed 30-epoch TRAIN-only formal experiment.
8. Only after the formal parameter Gate and checkpoint round trip PASS may an
   Exp12.5 supervised downstream comparison start.

Formal SSL uses the fixed final epoch, not VAL/TEST or proxy-task best selection.
The downstream comparison, if unlocked, must reuse the Exp12.3 supervised
split, epoch, batch, optimizer and seed protocol.
