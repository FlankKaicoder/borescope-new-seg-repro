# Exp10.1a seed43 Hard Treatment train-NaN root-cause probe

Status: **EXP10_DIAGNOSTIC_COMPLETE_WAITING_REVIEW**. Root-cause gate: **CASE C — FP32_MODEL_OR_OPTIMIZATION_INSTABILITY**. This was a TRAIN-only diagnostic; `val_accessed_for_probe=false` and `test_accessed=false`.

## Boundary and frozen inputs

The probe used seed43 baseline `best.pt` SHA256 `09f115c80de4d624f4fb36ee8ede1a65a372cce395d963a8e02e4b6bd65e732c`, the frozen 668-image TRAIN split, and the seed43 201-image hard pool. It did not run seed44, complete a new Treatment, evaluate VAL, access TEST, freeze a candidate, or start Exp11. No formal training hyperparameter was changed.

The requested/resolved Control and Treatment configurations match for checkpoint content, epochs=30, imgsz=640, batch=32, AdamW, lr0=0.01, lrf=0.01, weight decay, momentum/betas, warmup, augmentation, deterministic mode, seed=43, workers=4, sampler class, replacement, num_samples=668, dataset, TRAIN split, model content, and resume=false. The intended method difference is only sampling weights: Control all=1; Treatment normal=1/hard=2.

There is one secondary execution-fairness anomaly: the frozen Control and original Treatment logs show AMP self-check PASS/effective AMP=true, whereas the failed Treatment retry shows AMP self-check FAIL/effective AMP=false despite the same requested `amp=true`. This is recorded as `CONFIG_IMPLEMENTATION_BUG_CANDIDATE`, but exact replay shows it is not sufficient to explain the failure because both AMP and FP32 fail on the captured batch.

## Sampler and TRAIN integrity

- TRAIN: 668 images; hard=201; normal=467.
- Theoretical hard draw probability: `402 / 869 = 0.4626006904487917`.
- Deterministic epoch1 draw: 306 hard and 362 normal; actual hard ratio `0.45808383233532934`.
- `WeightedRandomSampler`, replacement=true, num_samples=668 in both arms; weights are applied once.
- Illegal indices=0, path mismatches=0, empty labels=0, wrong manifest=false; all pool rows carry seed43 and pool stems exactly match TRAIN.
- All 668 images decode and are finite. All labels parse; polygon coordinates are finite/in-range, class IDs valid, and no degenerate polygon/bbox or invalid target was found. Hard-pool invalid images/targets are also zero.

## First non-finite event

The effective retry path (`amp=false` after the logged engine self-check failure) reproduced immediately at epoch1, zero-based batch0 / one-based batch1, before optimizer step1. The 32-image augmented input is float32, shape `[32,3,640,640]`, range `[0,1]`, and finite; all targets are finite. Pre-step parameters/buffers are finite and the optimizer state is empty/finite.

Raw model output becomes non-finite during forward, before backward or any parameter update. Box, segmentation, classification, DFL, and total loss consequently become NaN. The event is therefore `FORWARD_NONFINITE`, not backward-, optimizer-, or update-originated.

The sampled stems are: `580, 33, 210, 69, 977, 451, 193, 287, 37, 675, 832, 54, 129, 44, 886, 623, 368, 327, 1001, 914, 579, 498, 26, 761, 901, 542, 958, 264, 291, 222, 444, 420`. Hard count=13, unique images=32, duplicate draws=0, maximum multiplicity=1. The largest original-label instance count in the batch is 28 (stem `33`). Per-sample hard score/error composition, instance count, and relative polygon area are in `first_nonfinite_samples.csv`.

## Exact replay and root-cause gate

The saved Treatment pre-step state and the separately saved baseline initial state differ in **0 tensors**: the first event occurred before any optimizer step, so the counterfactual states are identical. Four replays were run in independent fresh processes to avoid order effects:

| State | Mode | Raw output | Loss |
|---|---|---|---|
| Treatment pre-step | AMP | non-finite | non-finite |
| Treatment pre-step | FP32 | non-finite | non-finite |
| Baseline initial | AMP | non-finite | non-finite |
| Baseline initial | FP32 | non-finite | non-finite |

In isolated AMP replay, a unique early convolution (`model.1.conv`) is already non-finite before C2PSA. C2PSA qk and softmax are later non-finite, but are not the first event. In isolated FP32 replay, the first unique non-finite leaf is also in the early convolutional path (`model.1.conv` or `model.2.cv1.conv` across isolated deterministic-runtime repetitions); C2PSA qk matmul and softmax remain finite. This is not a recurrence of the earlier FP16-only C2PSA qk overflow waiver.

Therefore CASE B is rejected (`FP32` is not finite), CASE A is not supported (sampler/data/checkpoint mapping are valid and the AMP drift cannot explain an AMP-and-FP32 exact failure), and CASE D is rejected (the exact saved tensor/state reproduces in isolated processes). The applicable gate is **CASE C — FP32_MODEL_OR_OPTIMIZATION_INSTABILITY**, specifically an FP32 forward instability present before the first optimizer update.

## Artifacts

Tracked evidence is under `results/final_verify/exp10_1a_seed43_nan_probe/`:

- `config_diff_control_vs_treatment.txt`
- `sampler_audit.json`
- `epoch1_sample_sequence.csv`
- `train_data_integrity.csv`
- `first_nonfinite_batch.json`
- `first_nonfinite_samples.csv`
- `replay_amp_fp32.csv`
- `module_nonfinite_trace.csv`
- `root_cause_summary.json`

Large server-only artifacts are intentionally ignored by Git:

- `/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_1a_seed43_nan_probe/first_nonfinite_snapshot.pt`, SHA256 `fc1e78db6cb284f7ed3ebaefefa7351beda745ef001f8f01d71c10eff1b5e790`
- `/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_1a_seed43_nan_probe/baseline_initial_state.pt`, SHA256 `2c6a2e62d59717f35b0caa77ef13a7f894776055f2560ba5ec5b1fdb7509ec30`

Exp10 remains stopped and incomplete. Candidate Freeze and Exp11 remain forbidden pending review.
