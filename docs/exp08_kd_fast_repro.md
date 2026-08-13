# Exp08 classifier-teacher KD fast reproduction

Status: **SKIPPED_BY_ENGINEERING_GATE**. `test_accessed=false`.

This is a historical-method reconstruction, not a restoration of original code. The teacher is the frozen Exp06 SupCon ResNet18 checkpoint with SHA256 `8e22f17c029eb0f3cb9416673a3503e0d37c7b98b91f126da2107e23fe58c32b`. The implementation puts the teacher in evaluation mode and disables gradients. The student starts from official `yolo11n-seg.pt`.

Graph inspection selected layer 4 (`C3k2`): stride 8, 128 channels, and `[B,128,80,80]` at image size 640. Ground-truth boxes map to P3 ROIAlign 7x7, pooling, and a seven-class auxiliary head. Teacher ROIs use 1.2x boxes resized to 224x224 with ImageNet normalization; KD uses only the renormalized seven defect logits.

Smoke testing found and fixed a finite-check implementation bug. Online teacher ROI extraction then exhausted memory at batch 4; chunking teacher crops by 16 made batch 32 forward/loss finite, but one training step exceeded 70 seconds. Completing both 100-epoch runs would require a teacher-target cache or data-loader redesign outside the Fast Repro boundary. Under task rule 5.3, AUX_CE and KD formal training, VAL comparison, lambda/T search, and checkpoint selection were not run. The test split was never accessed.
