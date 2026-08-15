# Exp10.1b TRUE-FP32 review handoff

## Review decision

Exp10.1b completed the authorized TRAIN-only diagnostic. The final human interpretation is `EXP10_1A_ROOT_CAUSE=NONREPRODUCIBLE_DIAGNOSTIC_REPLAY`.

This does **not** mean the preprocessing tensor differed. Saved and recreated Trainer batches are hash-identical. It means the prior Exp10.1a non-finite replay evidence cannot be reproduced: the old replay code, the new explicit TRUE-FP32/TRUE-AMP replay, and formal Trainer no-update path now all return finite values with identical corresponding losses.

## Facts for reviewer

- Old FP32-labelled replay did not call `model.float()` or `input.float()`, but measured state/input were already entirely FP32; `EXP10_1A_FP32_REPLAY_INVALID_BY_DTYPE=false`.
- Serialized checkpoint EMA is half; `YOLO(checkpoint)` and Trainer setup both yield an FP32 master model (291 FP32 parameters, 180 FP32 floating buffers).
- TRUE-FP32 Gate passed and raw output/all losses were finite.
- TRUE-AMP from the same FP32 master state was finite.
- Formal Trainer-path AMP and TRUE-FP32 were finite and exactly matched standalone losses.
- `model.1.conv` was finite; conservative accumulation scale was about `9.60e3`.
- CPU control was not run because its prerequisite—GPU TRUE-FP32 non-finite—was false.
- No optimizer update, training epoch, VAL, TEST, seed44, Candidate Freeze, or Exp11 occurred.

## Required reviewer decision

The controlled Exp10 restart is now separately authorized under `docs/exp10_controlled_restart.md`; its strict ordered Gates apply.

Until that decision:

- Exp10 remains incomplete and stopped.
- Do not run seed44 or resume formal Treatment.
- Do not freeze a candidate.
- Do not enter Exp11 or access TEST.

Detailed evidence: `docs/exp10_1b_true_fp32_verification.md` and `results/final_verify/exp10_1b_true_fp32/root_cause_revised.json`.
