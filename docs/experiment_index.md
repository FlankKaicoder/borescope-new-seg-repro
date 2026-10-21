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


## Exp13 extension

| Experiment | Status | Canonical evidence |
|---|---|---|
| Exp13.1-R2 | DIAGNOSIS_COMPLETED_WAITING_REVIEW / 48 outputs | `docs/exp13_1_r2_controlled_generation_diagnosis.md` |

## Exp13 subsequent quality decision (2026-10-21)

The Exp13.1-R2 waiting-review row above is retained as a historical execution snapshot. The following row records the later authorized archival decision.

| Record | Status | Canonical evidence |
|---|---|---|
| Exp13.1-R2-QA | EXP13_1_R2_QUALITY_GATE_ARCHIVED / GENERIC_TEXT_ONLY_INPAINTING_NOT_APPROVED_FOR_CURRENT_SSL_SYNTHESIS | [Final quality review](exp13_1_r2_final_quality_gate_review.md) |

Current text-only inpainting SSL, seven-class generation, R3 parameter sweep, and Exp13.2 generation authorizations are all NO. Only Exp13.2-A method research/design discussion is the proposed next stage. Evidence is earlier ChatGPT qualitative sheet review relayed and authorized by the user; no new visual review or quantitative scores were produced. See [QA handoff](exp13_1_r2_qa_archival_handoff.md).

## Exp13.2-A0 historical restoration preparation (2026-10-21)

| Record | Status | Canonical evidence |
|---|---|---|
| Exp13.2-A0 | RESTORATION_REVIEW_ASSETS_PREPARED_AWAITING_HUMAN_REVIEW | [Preparation report](exp13_2_a0_restoration_feasibility_preparation.md); `results/exp13_restoration_feasibility/a0_20261021_081812/summary.json` |

88 historical candidates form 20 source groups; R2 sources overlap R1. Twenty source sheets and two class overviews are ready, with all human review fields blank. Historical crop truncation and nonzero mask-exterior differences remain review limitations. This later preparation record preserves all historical Gates and does not authorize quality PASS, new generation, training or Exp13.2-B. See [A0 handoff](exp13_2_a0_restoration_feasibility_handoff.md).

## Exp13.2-A0-QA source visual screening (2026-10-21)

| Record | Status | Canonical evidence |
|---|---|---|
| Exp13.2-A0-QA | EXP13_2_A0_SOURCE_VISUAL_SCREENING_ARCHIVED | [Screening review](exp13_2_a0_source_visual_screening_review.md); `results/exp13_restoration_feasibility/qa_20261021_090000/summary.json` |

The source-level first pass classified 15 sources as promising for further review, 2 uncertain, and 3 restricted. All 88 variant rows remain unaccepted: two qualitative notes for Crack 111 require follow-up, and 86 are pending. No trusted pair set is frozen; A1 discussion may continue, but Exp13.2-B/SSL remain unauthorized. See [QA handoff](exp13_2_a0_source_visual_screening_handoff.md).
