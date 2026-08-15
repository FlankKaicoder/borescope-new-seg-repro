# Exp10 controlled restart

Status: **COMPLETE / HARD_MINING_NOT_CONFIRMED / WAITING CANDIDATE FREEZE REVIEW**. `test_accessed=false`; Exp11 is forbidden.

## Final result

Seed43 Hard Treatment controlled restart passed 30/30 with all TRAIN losses finite, requested/effective AMP=true, FP32 master params/buffers, exact preflight/formal first-batch match, 20,040 sampled images, 331 optimizer steps, and finite/reloadable best/last checkpoints. Best SHA256: `8a97ebdf1a84ba43adb9fd7a4ea25d8990b7a81daee85b9a0b08bd5b69f62c69`.

Seed44 Baseline passed 100/100; best SHA256 `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`. Its TRAIN-only hard pool contains 201/668 images. Uniform Control and Hard Treatment both passed 30/30 from that same checkpoint, consumed 20,040 images and 331 optimizer steps, and have best SHA256 values `0c9eec4b3588dbe6b428641d70414d53776dd6cbb0c730103ede1cf381187671` and `af7d614718958cbc839d2a24a2416d660375cd7c3a1912d90c81af136aab6207`.

Unified VAL was rerun for all nine seed×model checkpoints. Treatment-Control Mask mAP50-95 deltas are seed42 `+0.030354`, seed43 `-0.006673`, and seed44 `-0.040311`; positive seed count is 1/3 and mean paired delta is `-0.005543 ± 0.035346` sample std. Final recommendation: `HARD_MINING_NOT_CONFIRMED`. Candidate Freeze recommendation is Baseline, but Candidate Freeze was not executed.

## Reason for restart

The historical seed43 formal Treatment run remains a real failed run. Exp10.1b established that the later diagnostic replay used to claim deterministic FP32/model-forward instability is non-reproducible. Current interpretation:

`EXP10_1A_ROOT_CAUSE=NONREPRODUCIBLE_DIAGNOSTIC_REPLAY`

The old failed output directories and evidence remain immutable. The restart uses a new independent directory.

## Frozen seed43 inputs

- Baseline best SHA256: `09f115c80de4d624f4fb36ee8ede1a65a372cce395d963a8e02e4b6bd65e732c`
- Frozen TRAIN-only hard pool: 201/668 images
- Hard-pool CSV SHA256: `645ef8731c721b5dbdb9da596e839185db6c6c72021acc7f8caa56b878e8f6d9`
- Split manifest SHA256: `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`
- Treatment weights: normal=1, hard=2; replacement sampler; 668 samples per epoch
- Formal configuration remains AMP=true, imgsz=640, batch=32, AdamW, deterministic=true, seed=43, epochs=30.

## Controlled numerical protocol

The protocol changes no training hyperparameter or mathematical definition. It adds only verification, recording, and fail-fast behavior:

1. Fresh-process TRAIN-only preflight from the frozen starting checkpoint and sampler first batch.
2. Require requested AMP=true and effective AMP=true.
3. Require every floating model parameter and buffer to be FP32 after Trainer setup.
4. Require formal first-batch stems and post-preprocess tensor SHA256 to match preflight exactly.
5. Require finite TRAIN forward/loss before backward; the first non-finite result raises a Hard Gate before backward/optimizer for that batch.
6. Disable Ultralytics NaN recovery/retry; reject any batch-size change.
7. Audit each saved checkpoint for load success, finite state tensors, and YOLO reload success.

If seed43 Treatment repeats TRAIN non-finite, no retry or new diagnosis is allowed. Exp10 stops as `HARD_MINING_FINAL_VERIFY_FAILED_NUMERICAL_ROBUSTNESS`; seed44 becomes `NOT_RUN_BY_GATE`.

All ordered Gates passed and Exp10 is complete. Candidate Freeze remains recommendation-only; TEST and Exp11 remain forbidden.
