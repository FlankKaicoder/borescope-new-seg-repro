# Method reconstruction status

| Method | Status | Interpretation |
|---|---|---|
| YOLO11n-seg Baseline | FINAL_SELECTED_METHOD | Frozen seed44 evaluated on TEST. |
| Low confidence | POSITIVE_DIAGNOSTIC / NOT_FINAL_MODEL | FN recovery signal with FP tradeoff. |
| Crack one-class | NO_CLEAR_GAIN | Diagnostic did not establish a gain. |
| Hard Mining | HARD_MINING_NOT_CONFIRMED | +0.030354/-0.006673/-0.040311; mean -0.005543. |
| ROI ResNet18 CE | COMPLETE_DIAGNOSTIC | ROI classification domain. |
| ROI CE + SupCon | POSITIVE_ROI_REPRESENTATION / NOT_FINAL_SEGMENTATION_METHOD | Macro F1 +0.015894. |
| Stage2 | NEGATIVE | Reconstructed system did not beat single-stage baseline. |
| KD | SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED | No formal performance claim. |
| SimSiam | INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED | 0/120 trainable backbone parameters changed. |
| Uniform continued training | CONTROL_ONLY / NOT_FINAL_CANDIDATE | Budget control; no post-hoc promotion. |
| Resolution | NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE | Future work only. |
