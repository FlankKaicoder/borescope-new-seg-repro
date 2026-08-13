# Exp09 SimSiam fast reproduction

Final status: **REAL_TRANSFER_FAILURE**. `test_accessed=false`.

The frozen SSL run remains valid as an execution result: 668 TRAIN images only, 100/100 epochs, batch 32, finite loss/checkpoint, and no representation-collapse signal. It was not rerun in Exp09.2a.

The original native-checkpoint byte gate was over-sensitive because Ultralytics 8.4.117 saves models in FP16. Revised verification established expected/loaded=240/240, missing/unexpected/shape mismatch=0, immediate trainable transfer=120/120, FP32 round-trip=240/240, and native FP16-explained round-trip=240/240. The old 80 mismatches were exactly BN running-mean/running-var buffers.

However, comparison with official COCO initialization found that the frozen SSL export changed 0/120 trainable backbone parameters and changed only 120 BN buffers. It therefore cannot prove learned SimSiam weights were transferred. The revised gate is `REAL_TRANSFER_FAILURE`; downstream 100-epoch training and VAL comparison are `NOT_RUN_BY_GATE`. See `docs/exp09_transfer_verification_repair.md`.
