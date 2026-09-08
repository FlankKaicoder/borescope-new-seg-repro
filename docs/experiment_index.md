# Experiment index

| Experiment | Final status | Primary evidence |
|---|---|---|
| Exp00 | PASS / historical Gates retained | `docs/01_dataset_audit.md` |
| Exp01 | DATASET_FREEZE_PASS | `docs/exp01_2_dataset_freeze.md` |
| Exp02 | PASS_WITH_NUMERICAL_WAIVER | `docs/exp02_final_handoff.md` |
| Exp03 | POSITIVE_DIAGNOSTIC | `docs/exp03_low_conf_fast_repro.md` |
| Exp04 | NO_CLEAR_GAIN | `docs/exp04_crack_oneclass_fast_repro.md` |
| Exp05/Exp10 | HARD_MINING_NOT_CONFIRMED | `results/final_verify/exp10_paired_deltas.csv` |
| Exp06 CE | COMPLETE_DIAGNOSTIC | `docs/exp06_roi_resnet_fast_repro.md` |
| Exp06 SupCon | POSITIVE_ROI_REPRESENTATION | `docs/exp06_supcon_fast_repro.md` |
| Exp07 | NEGATIVE | `docs/exp07_stage2_fast_repro.md` |
| Exp08 | SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED | `docs/exp08_kd_fast_repro.md` |
| Exp09 | INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED | `docs/exp09_simsiam_fast_repro.md` |
| Exp10 | COMPLETE | `results/final_verify/exp10_three_seed_summary.csv` |
| Exp11 | PASS / ONE_FINAL_FROZEN_EVALUATION | `results/final_test/exp11_final_result.json` |
| Exp12.0 | PASS / NO_TRAINING / WAITING_EXP12.1-S_AUTHORIZATION | `docs/exp12_0_runtime_probe.md` |
| Exp12.1-S | PASS / BACKBONE_UPDATE_CONFIRMED / WAITING_FORMAL_SSL_AUTHORIZATION | `docs/exp12_1s_smoke.md` |
| Exp12.1 | PASS / 100-EPOCH TRAIN-ONLY FORMAL SSL | `docs/exp12_1_formal_and_exp12_2_audit.md` |
| Exp12.2 | PASS / BACKBONE_UPDATE_CONFIRMED / WAITING_EXP12.3_AUTHORIZATION | `results/exp12_simsiam_basic/formal_20260822T071127Z/parameter_update_report.json` |

| Exp12.3-S | PASS / DOWNSTREAM FAIRNESS AND TRAINER-INIT GATE | results/exp12_downstream_ab/smoke_20260822T085916Z/smoke_gate.json |
| Exp12.3 | PASS / SINGLE-SEED FROZEN-VAL POSITIVE SIGNAL | docs/exp12_3_downstream_ab.md |
## Learning/report layer

The post-project Chinese walkthrough and experiment companions are indexed at `docs/zh/README.md`. This is a documentation and visualization layer, not a new experiment, and it does not alter any status above.

Exp12 is a separately scoped post-project research extension. It does not alter the Exp00–Exp11 final statuses or selected Baseline.


## Exp12 continuation

| Experiment | Status | Canonical evidence |
|---|---|---|
| Exp12.4-S | PASS / MULTI-SCALE LOCAL-CHANGE BACKBONE UPDATE GATE | `results/exp12_local_change/smoke_20260823T064652Z/summary.json` |
| Exp12.4 | PASS / 30-EPOCH TRAIN-ONLY LOCAL-CHANGE SSL | `docs/exp12_4_5_local_change.md` |
| Exp12.5-S0 | HARD_GATE / YAML PATH NORMALIZATION ONLY / RETAINED | `results/exp12_local_change_downstream/smoke_20260823T070045Z/smoke_gate.json` |
| Exp12.5-S | PASS / ROUTE C FAIRNESS AND TRAINER-INIT GATE | `results/exp12_local_change_downstream/smoke_20260823T070449Z/smoke_gate.json` |
| Exp12.5 | PASS / NO SINGLE-SEED FROZEN-VAL MASK GAIN | `docs/exp12_4_5_local_change.md` |

## Phase C1 Local Change review

| Experiment | Status | Canonical evidence |
|---|---|---|
| Phase C1 Quality Review | INSUFFICIENT_EVIDENCE_NEEDS_MANUAL_REVIEW | `docs/phaseC1_quality_review.md` |
| Phase C1 Manual Review Preparation | PREPARED_AWAITING_HUMAN_ANNOTATION | `docs/phaseC1_manual_review_protocol.md` and `results/phaseC1_manual_review/manual_review_preparation_manifest.json` |
| Phase C1-R1 Local Change V2 Repair | GENERATION_PASS / AUDIT_CONDITIONAL_PASS / C_D_IMPROVED | `docs/phaseC1_R1_repair_audit.md` and `results/phaseC1_repair_generation/phaseC1_r1_audit_20260908T100000Z/phaseC1_distribution_audit_report.json` |
