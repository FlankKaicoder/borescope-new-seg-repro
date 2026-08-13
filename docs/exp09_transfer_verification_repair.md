# Exp09.2a SimSiam backbone transfer verification repair

Final transfer-mechanism result: **PASS_REVISED**. Final Exp09 SSL status: **INVALID_BY_BACKBONE_NO_UPDATE**. Downstream training and VAL: **NOT_RUN_BY_GATE**. `test_accessed=false`.

## Why the old byte comparison was misleading

Ultralytics 8.4.117 was inspected directly in the project environment. `Model.save()` calls `deepcopy(self.model).half()` at `engine/model.py:364`; `Trainer.save_model()` serializes a half-precision EMA at `engine/trainer.py:725`; and `strip_optimizer()` calls `x["model"].half()` at `utils/torch_utils.py:828`. Raw FP32 byte hashes therefore cannot be the sole native-checkpoint criterion. Site-packages were not modified.

The old 80 byte mismatches were exactly 40 BN `running_mean` and 40 BN `running_var` buffers. None were trainable parameters.

## Revised transfer verification

- Key/shape: expected=240, loaded=240, missing=0, unexpected=0, shape mismatch=0.
- Immediate in-memory load before forward: all 120/120 trainable parameters matched the export after dtype normalization.
- FP32 state-dict round-trip: 240/240 PASS.
- Native Ultralytics round-trip: 240/240 explained by FP16 save conversion.
- Eval forward: zero BN buffer changes.
- Isolated train-mode diagnostic forward: 40 running means, 40 running variances, and 40 batch counters changed; this diagnostic copy did not contaminate downstream initialization.

The transfer mechanism is therefore verified. The decisive separate audit found that, compared with official COCO initialization, 0/120 trainable backbone parameters changed while 120/120 BN buffers changed. The SSL implementation/optimization did not update trainable YOLO backbone parameters. Consequently the SSL result is invalid for downstream adaptation, and performance is not evaluated. SSL and downstream were not rerun.
