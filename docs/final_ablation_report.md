# Final ablation and evidence report

## A. Dataset / baseline

969 supervised images and 1847 polygon instances span 7 classes. The global imbalance is 20.83:1 (corrosion 729 vs Tip curl 35). The frozen group-aware split contains 668/154/147 train/val/test images and zero near-duplicate cross-split leakage. YOLO11n-seg Baseline is the final selected segmentation method.

## B. Diagnostic findings

Exp03 recovered 91/173 VAL false negatives (52.6%) at lower confidence, but extreme lowering caused an FP explosion; this is a positive diagnostic, not a final model. Exp04 Crack one-class training showed no clear gain. Error evidence implicates confidence, localization, class confusion, imbalance and their coupling with object scale.

## C. Formal comparable segmentation results

Exp10 used identical VAL semantics and budget-matched controls. Hard Mining minus Uniform Control Mask mAP50-95 deltas were +0.030354, -0.006673 and -0.040311 for seeds 42/43/44; mean -0.005543, sample std 0.035346, only 1/3 positive. Therefore the seed42 preliminary gain was not reproduced and Hard Mining is `NOT_CONFIRMED`. Uniform continued training remains a control and cannot be promoted post hoc.

## D. Representation experiments

ROI ResNet18 CE reached accuracy 0.635135, macro F1 0.677804 and weighted F1 0.632591. SupCon reached macro F1 0.693698 (+0.015894). This supports improved ROI representation only; it is not evidence that segmentation improved.

## E. System experiments

The reconstructed YOLO + ROI classifier Stage2 system was negative: its fixed-point result did not exceed the single-stage baseline.

## F. Engineering-gated experiments

KD was not formally evaluated because the online-teacher path failed the engineering-cost Gate. It must not be described as a failed method.

## G. Invalid reconstruction

SimSiam remained finite/no-collapse, and the transfer mechanism later passed, but 0/120 trainable YOLO backbone parameters changed relative to COCO. The reconstruction is invalid for downstream claims; downstream was not evaluated.

## H. Multi-seed final verification

Baseline three-seed VAL Mask mAP50-95 was 0.298981, 0.299506 and 0.325157. Seed44 was selected before TEST using the highest frozen VAL score. Seed sensitivity is material, and single-seed positive ablations are insufficient.

## I. Final TEST

The frozen seed44 Baseline produced Box P/R/mAP50/mAP50-95 = 0.662385/0.474927/0.541636/0.293253 and Mask = 0.680727/0.498654/0.582704/0.271621 on 147 images/285 instances. Frozen VAL-to-TEST Mask mAP50-95 delta was -0.053536. This is a generalization observation only; no tuning or reselection followed.

## J. Limitations / Future Work

Resolution is `NOT_FORMALLY_ANSWERED`. Higher-resolution training was not prioritized because the frozen size audit did not show a simple monotonic smaller-object-worse-recall relation and the project prioritized broader reconstruction under limited compute/time. A preregistered 960 study and continued-fine-tuning study are optional future work, not unfinished work.
