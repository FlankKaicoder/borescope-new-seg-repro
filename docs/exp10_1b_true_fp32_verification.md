# Exp10.1b TRUE-FP32 / Trainer-path verification

## Conclusion

Exp10.1b is complete as a TRAIN-only, forward-and-loss, no-update diagnostic. The final human interpretation is:

`EXP10_1A_ROOT_CAUSE=NONREPRODUCIBLE_DIAGNOSTIC_REPLAY`

This does not claim that a preprocessing bug was found: the saved post-preprocess tensor and the deterministically recreated Trainer batch match in shape, statistics, every recorded tensor SHA256, and finiteness.

The earlier Exp10.1a `CASE C FP32_MODEL_OR_OPTIMIZATION_INSTABILITY` conclusion is superseded. In particular, “optimization instability” is not supported because the event was reported before optimizer step 1 and no optimizer step was performed in this verification.

## Scope and frozen inputs

- Frozen snapshot: `first_nonfinite_snapshot.pt`
  - SHA256: `fc1e78db6cb284f7ed3ebaefefa7351beda745ef001f8f01d71c10eff1b5e790`
- Frozen baseline initial state: `baseline_initial_state.pt`
  - SHA256: `2c6a2e62d59717f35b0caa77ef13a7f894776055f2560ba5ec5b1fdb7509ec30`
- Seed43 baseline best checkpoint SHA256: `09f115c80de4d624f4fb36ee8ede1a65a372cce395d963a8e02e4b6bd65e732c`
- Dataset: `/root/autodl-tmp/borescope-new-seg-data/v1`
- Only the saved exact augmented TRAIN batch, exact targets, and sample sequence were used for the formal replay.
- No optimizer step, epoch training, VAL, TEST, seed44, Candidate Freeze, Exp11, or formal Treatment resume was performed.

## Exp10.1a dtype-method audit

The old FP32-labelled branch performed `autocast(enabled=False)` but did not explicitly call `model.float()` or `input.float()`.

Despite that implementation omission, the frozen replay state actually contained 291 FP32 parameters and 180 FP32 floating buffers, with no FP16 or other floating dtypes. The saved input was also `torch.float32`. The model was in train mode on `cuda:0`, with autocast disabled.

Therefore:

`EXP10_1A_FP32_REPLAY_INVALID_BY_DTYPE=false`

The old result is not rejected merely because of parameter/input dtype.

## Checkpoint dtype transition

| Stage | FP16 params | FP32 params | FP16 buffers | FP32 buffers |
|---|---:|---:|---:|---:|
| Raw serialized checkpoint EMA | 291 | 0 | 180 | 0 |
| `YOLO(checkpoint)` load | 0 | 291 | 0 | 180 |
| Ultralytics Trainer setup complete | 0 | 291 | 0 | 180 |
| Explicit `model.float()` | 0 | 291 | 0 | 180 |

Direct measurement therefore confirms that the formal Ultralytics loading/training setup converts the serialized half checkpoint model back to FP32 master parameters and buffers.

## TRUE-FP32 Gate and result

The new replay explicitly executed `model.float()` and `batch["img"] = batch["img"].float()` before forward. It then verified all floating parameters and buffers were FP32, the input was finite FP32, all floating targets were finite, and CUDA autocast was disabled.

| Check | Result |
|---|---|
| Parameter dtype unique | `torch.float32` |
| Floating buffer dtype unique | `torch.float32` |
| Input dtype | `torch.float32` |
| Autocast | disabled |
| Raw output | finite |
| Box / seg / cls / DFL / semantic loss | all finite |
| Total loss | `236.9442596435547`, finite |
| First non-finite module | `NONE` |

Loss components were box `1.530516266822815`, seg `2.7600321769714355`, cls `1.883516550064087`, DFL `1.2304433584213257`, and semantic `0.0`.

## TRUE-AMP and formal Trainer-path controls

TRUE-AMP started from the same FP32 master state and FP32 input, then enabled CUDA autocast. Raw output and every loss were finite; total loss was `236.7904052734375`.

The formal Ultralytics Trainer setup reported 291 FP32 parameters, 180 FP32 floating buffers, and effective AMP `true`. The frozen post-preprocess batch was passed directly—without a second `/255`, normalization, cast, resize, or augmentation. Both formal Trainer-path branches were finite:

- AMP total loss: `236.7904052734375`
- Explicit TRUE-FP32 total loss: `236.9442596435547`

These values exactly match the standalone TRUE-AMP and TRUE-FP32 results.

## Preprocessing equivalence

The saved tensor is already the exact post-preprocess tensor. Both saved and recreated batches have image shape `[32, 3, 640, 640]`, dtype FP32, min `0`, max `1`, mean `0.3467880785`, std `0.2201775312`, and image tensor SHA256 `ecc715b97a4dcab4a9d55b310290aafe37b8eaaf2f6aaa0faeea9bf89b9f6123`. Every recorded batch tensor hash matches.

Thus no extra preprocessing operation explains the old non-finite artifact.

## Legacy replay reproducibility and revised root cause

The frozen Exp10.1a `--replay-only-mode` implementation was rerun in fresh processes without any update:

- FP32: raw output and loss finite; total loss `236.9442596435547`.
- AMP: raw output and loss finite; total loss `236.7904052734375`.

The old implementation now agrees exactly with the new gated replays and formal Trainer path. Together with hash-identical preprocessing, this makes the prior non-finite replay evidence non-reproducible. The historical formal seed43 Treatment failure remains real, but its deterministic FP32/model-forward explanation is invalidated and requires one controlled formal restart.

## Conv FP32 sanity and CPU control

`model.1.conv` was finite in TRUE-FP32:

- input max abs: `92.16114044189453`
- weight max abs: `0.7236328125`
- output max abs: `187.26121520996094`
- kernel: `3x3`, input channels `16`, groups `1`
- conservative accumulation scale: `9603.478837609291`

This scale is far below FP32 overflow. Because GPU TRUE-FP32 was finite, the conditional CPU TRUE-FP32 control was correctly not executed.

## State after Exp10.1b

- Current phase: `EXP10_TRUE_FP32_DIAGNOSTIC_COMPLETE_WAITING_REVIEW`
- Exp10 remains incomplete and stopped.
- seed44 was not run.
- Formal Treatment was not resumed.
- `val_accessed_for_probe=false`
- `test_accessed=false`
- Candidate Freeze was not executed.
- Exp11 was not executed and remains forbidden.
- No formal training hyperparameter was modified.

The next action requires human/ChatGPT review of whether Exp10 should later be restarted under a newly unified numerical protocol. This experiment does not authorize that action.

## Evidence

Evidence is under `results/final_verify/exp10_1b_true_fp32/`, including dtype audits, checkpoint transitions, TRUE-FP32/AMP results, module trace, Trainer-path results, preprocessing hashes, Conv sanity, conditional CPU record, legacy replay reproducibility, and `root_cause_revised.json`.
