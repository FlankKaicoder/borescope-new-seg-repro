# 实验索引

| 编号 | 名称 | 状态 | 文档 |
|---|---|---|---|
| Exp00.0 | Environment audit | PASS / TRAINING_GATE_STOP | `docs/exp00_0_environment_audit.md` |
| Exp00.1 | JSON/schema/pair audit | PASS / DATA_GATE_STOP | `docs/exp00_1_schema_pair_audit.md` |
| Exp00.2 | Class/polygon/statistics audit | PASS / CONSISTENCY_GATE_STOP | `docs/exp00_2_class_polygon_audit.md` |
| Exp00.3 | Duplicate/near-duplicate audit | PASS / LEAKAGE_GATE_STOP | `docs/exp00_3_duplicate_audit.md` |
| Exp00.4 | Unpaired-image review package | PASS / HUMAN_DECISION_PENDING | `docs/exp00_4_unpaired_image_review.md` |
| Exp00.5 | Near-duplicate conflict review package | PASS / HUMAN_DECISION_PENDING | `docs/exp00_5_near_duplicate_conflict_review.md` |
| Env00.1 | Isolated training environment preparation | PASS / NO_TRAINING_STARTED | `training_environment_report.md` |
| Exp01.0 | JSON → YOLO segmentation conversion | PASS | `docs/exp01_0_json_to_yolo.md` |
| Exp01.1 | Group-aware stratified split | PASS / 0 LEAKAGE | `docs/exp01_1_group_aware_split.md` |
| Exp01.2 | Dataset v1 freeze and verification | PASS / DATASET_FREEZE_GATE_PASS | `docs/exp01_2_dataset_freeze.md` |
| Exp02.0 | YOLO11n-seg batch probe and one-epoch smoke | PASS / SMOKE_GATE_PASS | `docs/exp02_0_smoke_batch_probe.md` |
| Exp02.1 | YOLO11n-seg 640 / 100-epoch baseline | COMPLETE / PASS_WITH_NUMERICAL_WAIVER | `docs/exp02_1_yolo11n_seg_640_baseline.md` |
| Exp02.2 | Baseline size and error audit | PASS / VAL_ONLY | `docs/exp02_2_baseline_size_error_audit.md` |
| Exp02.2a | Early-validation-loss NaN root-cause probe | COMPLETE / CASE_C / WAIVER_ACCEPTED_LATER | `docs/exp02_2a_early_val_nan_root_cause.md` |
| Exp03 | Low-confidence recovery | COMPLETE / POSITIVE / VAL_ONLY | `docs/exp03_low_conf_fast_repro.md` |
| Exp04 | Crack one-class diagnostic | COMPLETE / NO_CLEAR_GAIN | `docs/exp04_crack_oneclass_fast_repro.md` |
| Exp05 | Fair hard-mining comparison | COMPLETE / POSITIVE_CANDIDATE | `docs/exp05_hard_mining_fast_repro.md` |

| Exp06.0/1 | ROI patch + ResNet18 CE | COMPLETE / VAL ONLY | `docs/exp06_roi_resnet_fast_repro.md` |
| Exp06.2 | ResNet18 CE + SupCon | COMPLETE / SUPCON_POSITIVE | `docs/exp06_supcon_fast_repro.md` |
| Exp07 | YOLO + ResNet Stage2 | COMPLETE / NEGATIVE / VAL ONLY | `docs/exp07_stage2_fast_repro.md` |

| Exp08 | Classifier Teacher to YOLO KD | SKIPPED_BY_ENGINEERING_GATE | `docs/exp08_kd_fast_repro.md` |
| Exp09/09.2a | SimSiam reconstruction and transfer verification | INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED | `docs/exp09_transfer_verification_repair.md` |
| Exp10 | Three-seed Baseline vs Hard Mining | STOPPED / SEED43 TREATMENT TRAIN NONFINITE | `docs/exp10_three_seed_final_verify.md` |
| Exp10.1a | Seed43 Treatment TRAIN NaN root-cause probe | COMPLETE / CASE C / WAITING REVIEW | `docs/exp10_1a_seed43_treatment_nan_root_cause.md` |
