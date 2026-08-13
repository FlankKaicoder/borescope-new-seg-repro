# Exp09 SimSiam fast reproduction

Final status: **INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED**. Downstream status: **NOT_RUN_BY_GATE**. `test_accessed=false`.

The frozen SSL execution remains recorded: 668 TRAIN images only, 100/100 epochs, batch 32, finite loss/checkpoint, finite feature and embedding standard deviations, and no collapse signal. It was not rerun or tuned during status calibration.

Exp09.2a established that checkpoint transfer is correct. Key/shape verification was 240/240 with no missing, unexpected, or shape-mismatched tensors; immediate trainable transfer was 120/120; FP32 state-dict round-trip was 240/240; and native Ultralytics round-trip was 240/240 after accounting for its FP16 save conversion. Eval forward changed zero BN buffers. The old 80 byte mismatches were exactly 40 BN running means and 40 BN running variances.

The decisive finding is upstream of transfer: relative to official COCO initialization, the SimSiam export changed 0/120 trainable backbone parameters and 120/120 BatchNorm buffers. This reconstruction therefore did not form effective trainable backbone adaptation. Without a valid adapted trainable backbone, downstream training and VAL comparison were forbidden, so SimSiam method performance is `NOT_EVALUATED`, not negative.
