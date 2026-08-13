# Exp09.2a SimSiam backbone transfer verification repair

Final gate: **REAL_TRANSFER_FAILURE**. `test_accessed=false`. Exp09 downstream training and VAL are `NOT_RUN_BY_GATE`.

## Why the old gate could misclassify

Ultralytics 8.4.117 was inspected directly from the project environment. `Model.save()` calls `deepcopy(self.model).half()` at `engine/model.py:364`; `Trainer.save_model()` serializes a half-precision EMA at `engine/trainer.py:725`; `strip_optimizer()` calls `x["model"].half()` at `utils/torch_utils.py:828`. Raw FP32 byte hashes therefore cannot be the sole native-checkpoint criterion. Site-packages were not modified.

The old 80 byte mismatches were exactly 40 BN `running_mean` and 40 BN `running_var` buffers. None were trainable parameters. Native round-trip was 240/240 exact after reproducing FP16 quantization; FP32 state-dict control was 240/240 exact.

## Revised transfer evidence

- Key/shape gate: expected=240, loaded=240, missing=0, unexpected=0, shape mismatch=0.
- Immediate in-memory load before any forward: all 120/120 trainable parameters exactly matched the SimSiam export after dtype normalization.
- Eval forward changed zero BN buffers.
- Diagnostic train-mode forward changed all 40 running means, 40 running variances, and 40 batch counters; this copy never contaminated formal initialization.
- The decisive failure: compared with the official COCO initialization, 0/120 trainable backbone parameters had changed. All 120 changed tensors were BN buffers. There is therefore no trainable parameter satisfying `COCO != SimSiam == downstream`.

The revised verifier proves that loading and checkpoint serialization work, but it also proves the frozen SSL export contains no learned trainable backbone update. Under the explicit Gate definition this is a real transfer failure, not a serialization false alarm. SSL was not rerun or tuned, and downstream training was forbidden.
