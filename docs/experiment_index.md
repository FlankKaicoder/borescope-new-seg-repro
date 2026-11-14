# Experiment index

## Exp14-A10 A dataset freeze (2026-11-14)

EXP14_A10_A_DATASET_FROZEN_WITH_EXCEPTIONS. 392 final pixel identities audited, 379 A images and labels frozen, 13 same-asset exceptions isolated. Fixed CPU parsing verified the 668-image baseline and 1,047-image augmentation arm. [Handoff](exp14/EXP14_A10_A_DATASET_FREEZE_HANDOFF.md) · [Sanitized summary](../results/exp14_a10_a_dataset_freeze/summary.json).

## Exp14-A9D1 full 392-source recovery (2026-11-14)

EXP14_A9D1_FULL392_RECOVERY_COMPLETE_PENDING_HUMAN_CONFIRMATION. 138 historical pixel bindings, 166 proposed recoveries, 79 user decisions and 9 proposed whole-image exclusions. The private Chrome-tested review page covers all 392 sources and binds user choices to exact pixels. [Handoff](exp14/EXP14_A9D1_FULL392_RECOVERY_HANDOFF.md) · [Sanitized summary](../results/exp14_a9d1_full392_recovery/summary.json).

## Exp14-A9-D0 admission diagnosis (2026-11-14)

EXP14_A9D0_DIAGNOSIS_PENDING_CHATGPT_GATE. A0 reproduced 392 → 200 with 192 attributable holds. A visual evidence covers 36 cases. B ROI support was screened on 545 archived candidates and visually checked in a 36-case regional G1 pilot. Formal admissions and Trusted Pairs remain zero. [Handoff](exp14/EXP14_A9_D0_ADMISSION_ROOT_CAUSE_REVIEW.md) · [Sanitized summary](../results/exp14_a9d0_admission_diagnosis/summary.json).

## Exp14-A9 final candidate consolidation (2026-11-14)

EXP14_A9_FINAL_DATASET_CONSOLIDATED_PENDING_CHATGPT_DATA_GATE. 523 TRAIN sources, 901 Polygons and 956 human review identities. The independently audited package contains 200 A suggestions and 545 B Polygon proposals across 376 sources. [Handoff](exp14/EXP14_A9_FINAL_DATASET_CONSOLIDATION_HANDOFF.md) · [Sanitized summary](../results/exp14_a9_final_dataset/summary.json).

## Exp14-A8 final V4 repair and registration (2026-11-13)

EXP14_A8_FINAL_V4_COMPLETE_PENDING_HUMAN_REVIEW. 31 unique V4 POSTs, 31 Raw/Aligned/Bounded images, 31 visual reviews and a cross-version ledger for 523 sources and 901 Polygons. The private final reviewer supports historical version retention. [Handoff](exp14/EXP14_A8_FINAL_V4_REPAIR_HANDOFF.md) · [Sanitized summary](../results/exp14_a8_final_v4/summary.json).

## Exp14-A7B third-round completion (2026-11-13)

EXP14_A7B_V3_COMPLETION_PENDING_HUMAN_REVIEW. A7 dispatch reached 94/94 with 82 new POSTs, 84 new Raw/Aligned/Bounded images and 84 AI visual reviews. Five original delivery failures remain locked. The supplemental Chrome-tested page has zero review IDs in common with the first batch. [Handoff](exp14/EXP14_A7B_V3_COMPLETION_HANDOFF.md) · [Sanitized summary](../results/exp14_a7b_v3_completion/summary.json).

## Exp14-A7 targeted repair and registration (2026-11-13)

EXP14_A7_V3_REPAIR_PARTIAL_PENDING_HUMAN_REVIEW. 12 unique POSTs, 7 new Raw images, 23 new Aligned candidates including 16 old Raw rescues, and 23 AI visual reviews. The relay stop rule left 82 frozen requests unsent. The private offline reviewer passed Chrome verification. [Handoff](exp14/EXP14_A7_TARGETED_V3_REPAIR_HANDOFF.md) · [Sanitized summary](../results/exp14_a7_targeted_v3/summary.json).

## Exp14-A6 targeted repair and registration (2026-11-13)

EXP14_A6_V2_REPAIR_COMPLETE_PENDING_HUMAN_REVIEW. 185 unique POSTs, 200 new Raw images, 194 newly generated and 88 rescued Aligned candidates. The private comparison page holds all 288 AI visual reviews. [Handoff](exp14/EXP14_A6_TARGETED_V2_REPAIR_HANDOFF.md) · [Sanitized summary](../results/exp14_a6_targeted_v2/summary.json).

## Exp14-A5 full AI visual review (2026-11-12)

EXP14_A5_AI_REVIEW_COMPLETE_PENDING_HUMAN_CONFIRMATION. Image-based suggestions cover 530/530 variants from 523 TRAIN sources, including 911 per-variant Polygon decisions. The offline assisted page separates AI drafts from formal human decisions. [Handoff](exp14/EXP14_A5_FULL_AI_VISUAL_REVIEW_HANDOFF.md) · [Sanitized summary](../results/exp14_a5_ai_review/summary.json).

## Exp14-A5 Pilot40 visual evidence (2026-11-12)

EXP14_A5_PILOT40_VISUAL_EVIDENCE_READY. Forty fixed variants from 39 TRAIN sources, 31 Aligned, 9 alignment-unverified and 66 selected Polygons. Two 20-case evidence ZIPs verified with blank AI suggestions and private historical controls. [Handoff](exp14/EXP14_A5_PILOT40_VISUAL_EVIDENCE_HANDOFF.md) · [Sanitized summary](../results/exp14_a5_pilot40/summary.json).

## Exp14 large-scale TRAIN repair and review (2026-11-12)

EXP14_LARGE_SCALE_REPAIR_COMPLETED_PENDING_HUMAN_REVIEW. Native-frame Raw coverage is 523 of 523 SELECT sources after 485 authorized POSTs and verified historical reuse. The offline bundle has 530 review entries and 379 Aligned candidates from 374 sources. New human approvals and Trusted Pairs are zero. [Handoff](exp14/EXP14_LARGE_SCALE_REPAIR_HANDOFF.md) · [Sanitized summary](../results/exp14_large_scale_repair/summary.json).

## Exp14-A4-R1 complete matched E0/E1 run (2026-11-11)

Both seed42 arms passed 100 epochs and 2,200 effective updates with zero skips and verified final checkpoints. Frozen VAL Mask mAP50–95 delta E1−E0 was −0.017197. No TEST evaluation. EXP14_A4_R1_PENDING_CHATGPT_RESULT_GATE. [Handoff](exp14/EXP14_A4_R1_FORMAL_E0_E1_REVIEW.md).

## Exp14-A4-D1 zero-gradient audit (2026-11-11)

Six reconstructed TRAIN batches compared under AMP Scale=1 and true FP32 at the A4 stopped model state. The stride-32 box and mask branches were zero in both precisions for the two foreground-empty cases and active for the other four. Zero parameter updates in diagnosis. PASS_CONDITIONAL_RESTART. [Handoff](exp14/EXP14_A4_D1_ZERO_GRADIENT_REVIEW.md).

## Exp14-A4 first formal paired run (2026-11-11)

Exp14-A4 formal E0 stopped at epoch 43 batch 2 on the predeclared three-batch zero-gradient rule. 42 complete epochs, 926 effective updates, zero native skips, finite checkpoint. E1 and VAL not started. EXP14_A4_PENDING_CHATGPT_RESULT_GATE. [Handoff](exp14/EXP14_A4_FORMAL_E0_E1_REVIEW.md).

## Exp14-A3-D2 revised AMP Smoke (2026-11-11)

Exp14-A3-D2 bounded Smoke passed both arms with native AMP init_scale=1. 2 epochs/44 batches/44 effective updates each, zero skips, matched slot and LR trajectories. One isolated E1 zero-gradient proportion spike is recorded. EXP14_A3_D2_PENDING_CHATGPT_FORMAL_TRAIN_GATE. No formal training or VAL/TEST. [Handoff](exp14/EXP14_A3_D2_SMOKE_REVIEW.md).

## Exp14-A3-D1 first-batch numerical diagnosis (2026-11-11)

Exp14-A3-D1 replayed the A3 first batch, matched its loss exactly and completed five backwards with zero updates. Scale 65536/4096/256 had nonfinite gradients; AMP scale 1 and FP32 were finite. Native GradScaler probe skipped the underlying optimizer and backed off 65536 to 32768. AMP_SCALE_OVERFLOW_LIKELY. Formal training remains stopped. [Handoff](exp14/EXP14_A3_D1_NUMERICAL_REVIEW.md).

## Exp14-A3 bounded AutoDL Smoke (2026-11-11)

Exp14-A3 AutoDL TRAIN-only preflight passed in both arms (684 samples, 22 batches, tail 12). E0 stopped at the first training batch on a non-finite AMP-scaled gradient before AdamW. Effective updates=0; E1 update phase not started; no VAL/TEST. SMOKE_STOP_PENDING_CHATGPT_REVIEW. [Handoff](exp14/EXP14_A3_SMOKE_HANDOFF.md).

## Exp14-A2 matched data freeze (2026-11-11)

Exp14-A2 verified 21 A/B-common sources, retained 16 and placed 5 on full-frame support HOLD. Frozen baseline=668 images/1,266 instances; aligned=16 explicit empty labels. E0/E1=684 draws per epoch and 2,200 planned updates each. 38 CPU tests passed. FROZEN_DATA_PENDING_CHATGPT_SMOKE_GATE. [Handoff](exp14/EXP14_A2_HANDOFF.md).

## Exp14-A1 human-review import (2026-11-11)

Exp14-A1 imported 24 human-reviewed TRAIN variants and verified 22 A candidates and 21 B Pair candidates. All selected final images are aligned. Fifteen variants remain outside priority review. CANDIDATES_VERIFIED_PENDING_CHATGPT_DATA_GATE. No training dataset released. Twenty-seven CPU tests passed. [Handoff](exp14/EXP14_A1_HANDOFF.md).

## Exp14-A0 pseudo-normal data admission audit (2026-11-11)

Exp14-A0 completed the offline file audit for 36 frozen TRAIN sources and 39 repair variants. Geometry candidates=29; A/B human pending=39 each; A/B approved=0; label plans on HOLD=44. TRAINING_DATA_NOT_YET_APPROVED. Eleven CPU synthetic tests passed. Stop for ChatGPT data Gate and human review. [Handoff](exp14/EXP14_A0_HANDOFF.md).

## Exp13-G3-D2 final bounded closeout (2026-11-09)

`G3D2_CLOSED_BOUNDED_PARTIAL_HUMAN_EVIDENCE`. Annotation ended by user choice:
120 preset slots,22 confirmed,15 rule-valid,7 NEAR-region exclusions,
98 pending,0 complete evaluable Pairs. Partial-point fixed-transform TRE is
exploratory; mechanical identity lacks independent expert verification.
`FORMAL_TRE_GATE=NOT_ESTABLISHED`; `TRUSTED_PAIR_PASS=0`. Retain the SIFT +
proportional mapping/Similarity/Affine baseline; no further point completion,
60-source generation or formal Local Change training follows. [Final report](exp13_cloud/G3_D2_FINAL_CLOSEOUT.md),
[handoff](exp13_cloud/G3_D2_HANDOFF.md), [sanitized metrics](../results/exp13_g3d2/closeout_partial_summary.json).
Stop for ChatGPT Gate and research-route decision. Earlier entries are historical snapshots.

## Exp13-G3-D2 independent mechanical benchmark and Pair quality (2026-11-09)

`G3D2_BENCHMARK_READY_PENDING_HUMAN_REVIEW_AND_CHATGPT_GATE`. Eighteen variants
from17 TRAIN sources cover six classes, holdout disagreements, UI/multi-Polygon
risks and C2 556/640/645. Offline native-coordinate landmark review has120
empty slots with JSON recovery. Saved C0/C2 and D1 refits are frozen separately;
the evaluator never fits manual points. Actual human TRE and18 fresh Pair
reviews remain PENDING, Trusted Pair PASS=0. D1 interpretation/test-count
errata are corrected; full-mask audit finds914 omitted441 pixels. D2 tests:
Python8/JS10/Chrome14 PASS; existing Python9/UI10 PASS. See [report](exp13_cloud/G3_D2_MECHANICAL_BENCHMARK_REVIEW.md),
[handoff](exp13_cloud/G3_D2_HANDOFF.md), [sanitized result](../results/exp13_g3d2/sanitized_summary.json).
Private atlas: `E:\borescope_review_assets\g3d2_mechanical_benchmark_20261109_release_verified\index.html`.
Stop for ChatGPT audit/human review; no60-source generation or training.

## Exp13-G3-D0/D1 geometry registration and Local Change Pair quality (2026-11-09)

`G3D1_METHOD_STUDY_COMPLETE_PENDING_CHATGPT_AUDIT_GATE`. Independent spatial
holdout evaluated 36 frozen C0 and 3 C2 V2 pairs with SIFT and ORB under fixed
Similarity/Affine RANSAC. C0 holdout: SIFT 16 PASS / 17 FAIL / 3 insufficient;
ORB 16 / 19 / 1. C2 645 passes both checks; 556 remains insufficient and
640 fails both on held-out tails. These do not change historical geometry
labels. Synthetic known-transform control error=0.078 px; Trusted Pair PASS=0.
Keep mature SIFT/resize/Similarity/Affine C0 as baseline and use holdout plus
manual mechanical/local review as additional Gate evidence. The 60-source
repair batch is not ready. See [research report](exp13_cloud/G3_D0_D1_ALIGNMENT_PAIR_QUALITY_RESEARCH.md),
[handoff](exp13_cloud/G3_D0_D1_HANDOFF.md), and private
`E:\borescope_review_assets\g3d1_alignment_method_study_20261109\alignment_validation.json`.

## Exp13-G3-C2 Prompt V2 preparation (2026-11-09)

`G3C2_V2_LIVE_READY`. The five frozen retry cases were diagnosed separately.
Four sources are proposed for at most four serial requests; the corrosion case
whose Polygon overlaps green UI is HOLD. Category Prompt V2 files and a new
attempt manifest are hash-pinned. Repair Masks are private byte-identical
copies of selected V1 Polygon masks because the evidence did not support a
safe expansion. Executor Dry-run: 4/4, POST=0, no authorization. Cost remains
`COST_UNKNOWN`; no V2 images, aligned outputs, composites, or Trusted Pairs
exist. See [preparation review](exp13_cloud/G3_C2_V2_PREPARATION_REVIEW.md).

## Exp13-G3-C1 human review closeout (2026-11-08)

`G3C1_VISUAL_REVIEW_CLOSEOUT_READY`. The recovered exact-set contains 31
visual passes and 5 retries. Of 31 passes, 24 have a geometry candidate and 7
remain alignment-unverified. One pass note rejects Pair use, leaving 23
unambiguous cases proposed for bounded Pair-fidelity review; Trusted Pair
PASS remains zero. A visual decision/notes discrepancy and raw-output UI
removal are disclosed without changing the CSV. See
[closeout](exp13_cloud/G3_C1_VISUAL_REVIEW_CLOSEOUT.md).

## Exp13-G3-C0 review UI recovery (2026-11-08)

`G3C0_REVIEW_UI_RECOVERY_PASS`. Restored the user's private 36-row review CSV
into a new offline page: 31 `VISUAL_PASS`, 5 `NEEDS_RETRY`, all six nonempty
notes preserved, and Geometry remains equal to frozen 28/8 diagnostics. Added
transactional CSV import, robust UTF-8-BOM export, delayed Blob URL release,
copy fallback, and guarded local autosave. Tests: Node 10/10 and Python 3/3.
See [handoff](exp13_cloud/G3_C0_REVIEW_UI_RECOVERY_HANDOFF.md).

## Exp13-G3-C0 batch alignment (2026-11-08)

`G3C0_BATCH_ALIGNMENT_READY_FOR_VISUAL_REVIEW`. A fixed CPU SIFT/RANSAC pass
processed all 36 saved G3-B2 TRAIN outputs after source/JSON/Mask/output
identity checks. 28 mappings passed the bounded geometry screen (4
SIFT-validated proportional resize, 5 Similarity RANSAC, 19 Affine RANSAC); 8
remain `ALIGNMENT_UNVERIFIED`. Native-size composites change only valid pixels
inside selected target masks. Visual decisions remain `UNREVIEWED`, and
Trusted Pair PASS=0. See [handoff](exp13_cloud/G3_C0_ALIGNMENT_HANDOFF.md).

## Exp13-G3-B2 first authorized batch generation (2026-11-08)

`G3B2_BATCH_GENERATION_COMPLETE_PENDING_VISUAL_REVIEW`. The user-authorized
36-source batch returned 36/36 HTTP 200 outputs with no retries, failures or
skips. Usage totaled 12,161 tokens; response cost was absent for all calls and
remains unknown. Provider output dimensions differ from native source sizes,
so no alignment or Trusted Pair claim is made. All private visual decisions
start `UNREVIEWED`. See [generation review](exp13_cloud/G3_B2_BATCH_GENERATION_REVIEW.md).

## Exp13-G3-B2-B batch executor preparation (2026-11-08)

`G3B2B_EXECUTOR_READY_FOR_HUMAN_REVIEW`. Expanded the private initial
proposal to 36 distinct TRAIN sources, six per class across Burn, Crack, Dent,
Material missing, Tears and corrosion. All review rows remain PENDING with no
paid authorization. The G3-specific Manifest-driven executor passed 9/9
synthetic Mock tests and a 36-source private Dry-run with zero POSTs. Provider
identity, resolution support and price are not independently established;
cost remains unknown. See [handoff](exp13_cloud/G3_B2_B_EXECUTOR_PREPARATION.md).

## Exp13-G3-B1 review export correction and emergency backup restore (2026-11-07)

`G3B1_REVIEW_EXPORT_FIX_READY`. SELECT requires explicit Polygon targets;
MAYBE allows none; REJECT exports no targets. A 668-row emergency JSON was
validated and restored to a new private CSV without overwriting the baseline.
The original HTML remains byte-identical and a separately named repaired page
is available. See [review export fix handoff](exp13_cloud/G3B1_REVIEW_EXPORT_FIX_HANDOFF.md).

## Exp13-G3-B1 complete TRAIN source-level human review preparation (2026-11-07)

`G3B1_HUMAN_SOURCE_REVIEW_READY`. The private offline source atlas contains one
card per 668 TRAIN sources and links all 1,266 Polygon targets to the preserved
instance-level atlas. Per-class source counts are Burn 141, Crack 80, Dent 120,
Material missing 154, Tears 45, Tip curl 27, and corrosion 195. Decisions start
`UNREVIEWED`; selected Polygon IDs are explicit and do not imply a Trusted
Training Pair. See [G3-B1 handoff](exp13_cloud/G3_B1_HUMAN_SOURCE_REVIEW_HANDOFF.md).

## Exp13-G3-A export integrity and G2 aggregation fix (2026-11-07)

Targeted fix prepared at the current Exp13 branch; awaiting independent GitHub
audit before any AutoDL export. Frozen Manifest SHA is verified independently;
the TRAIN-only Export Manifest records a one-to-one source association and
source/export hashes. ZIP contains copied files, both manifests, class mapping,
and verification records. Windows exact-set/hash/JSON/668-image/1,266-instance
verification gates the formal private atlas. G2 source issues now force a
top-level failure; cost and upstream identity unknowns remain separate from
execution integrity. Synthetic regression suite 10/10 PASS; private G2 CPU
re-audit `PASS_WITH_LIMITATIONS` with zero source issues. See
[fix review](exp13_cloud/G3_A_EXPORT_FIX_REVIEW.md).

Latest extension: **Exp13.2-H1 — `EXP13_2_H1_COMPLETED_PENDING_CHATGPT_GATE`**. [Review](exp13_2_h1_hallucination_replay_review.md); [handoff](exp13_2_h1_handoff.md); `results/exp13_2_h1_hallucination_replay/20261026T075800Z_h1/`. One exact source23/R2 failure, existing DS reused, one SDXL512 success/0retry. Old generation prompt retained: no pseudo-normal efficacy or model-ranking claim. Structural hallucination suspected, human decision pending; production/trusted pairs0. Successor cumulative17, Smoke4/4 remaining0, aggregate103/all-local283. Controlled17PNG ZIP7170070bytes SHA2569b7251f3470ceef431287d0878d2ec9fb04fa52c3c852268ad1a36194c2bf9e8; CPU validation PASS technical only; no VAL/TEST/raw writes/training. Stop for Gate.

Latest extension: **Exp13.2-M1 — `EXP13_2_M1_COMPLETED_PENDING_CHATGPT_GATE`**. [Review](exp13_2_m1_execution_review.md); [handoff](exp13_2_m1_handoff.md); `results/exp13_2_m1_matched_restoration_benchmark/20261026T064000Z_m1/`. User input PASS and12-call authorization recorded in a new overlay;12success/0failure/0retry,6complete pairs. CPU integrity/budget/blind-package checks PASS technical only. Six-sheet anonymous ZIP is controlled; semantic quality and winner PENDING. Smoke4/4 remaining0; aggregate104/all-local284; no trusted pair, VAL/TEST, raw writes or training. Await ChatGPT/user review; no automatic continuation.

Latest extension: **Exp13.2-M0-R1 — `EXP13_2_M0_R1_PREPARED_AWAITING_HUMAN_INPUT_REVIEW`**. [Review](exp13_2_m0_r1_source_reselection_review.md); [handoff](exp13_2_m0_r1_handoff.md); `results/exp13_2_m0_r1/20261025T122430Z_m0r1/`. Retained469/260/470/8 defaults verified; new corrosion1/2 reach preferred3+3 geometry pool after two fixed-order inspections. All6 user input decisions pending. Controlled62,181,247-byte ZIP contains16 sheets/10 alternatives, exact512/1024/effective masks and pending template;230 entries verified. Proposed future calls12, executable0; S5 remains4/4 with0 Smoke, no model/inference/training. Exclusions9/23 and history intact; no M1 authorization.

Latest extension: **Exp13.2-M0 — `EXP13_2_M0_SOURCE_OR_PROTOCOL_HOLD` / `SOURCE_SELECTION_HOLD`**. [Review](exp13_2_m0_matched_comparison_preflight.md); [handoff](exp13_2_m0_handoff.md); `results/exp13_2_m0_matched_model_benchmark/20261025T115110Z_m0/`. Six TRAIN sources audited CPU-only, four geometry-only candidates (Crack3/corrosion1), zero approved inputs. Shared512/1024 protocol, static local model hashes and future blind review preregistered; no model/inference/training or M1 authorization. S5 remains4/4.

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

## Exp13.2-A1 method design (2026-10-21)

| Record | Status | Canonical evidence |
|---|---|---|
| Exp13.2-A1 | EXP13_2_A1_DESIGN_READY_FOR_CHATGPT_REVIEW | [Design specification](exp13_2_a1_method_design_spec.md), [code audit](exp13_2_a1_code_reuse_audit.md), [fairness protocol](exp13_2_a1_fairness_protocol.md), [future gates](exp13_2_a1_gate_definition.md), [handoff](exp13_2_a1_design_handoff.md) |

Documentation-only design and code-surface audit. No model, data, training code, or historical result changed. Design is not frozen; no trusted pairs are frozen; Restoration Quality Gate, Exp13.2-B, and SSL remain unauthorized.

## Exp13.2-A1-R1 design consistency repair (2026-10-23)

| Record | Status | Canonical evidence |
|---|---|---|
| Exp13.2-A1-R1 | EXP13_2_A1_R1_DESIGN_READY_FOR_CHATGPT_REVIEW | [Review response](exp13_2_a1_r1_design_review_response.md); [handoff](exp13_2_a1_r1_handoff.md); revised A1 method/fairness/gate documents |

Document-only repair of loss regions, T2/T3 matching, proposed budgets and artifact gates. Previous A1/A0 entries remain historical snapshots. Trusted-pair freeze, Restoration Quality Gate PASS, Exp13.2-B and SSL authorizations remain NO; no runtime experiment or registry change.

## Exp13.2-A2 Batch 01 human decision registration and spatial audit (2026-10-23)

| Record | Status | Canonical evidence |
|---|---|---|
| Exp13.2-A2-Batch01-UserReview | EXP13_2_A2_BATCH01_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT | [User decision record](exp13_2_a2_batch01_human_review_decision.md); [spatial audit](exp13_2_a2_batch01_spatial_audit.md); [handoff](exp13_2_a2_batch01_handoff.md); [overlay](../results/exp13_restoration_feasibility/qa_20261023_batch01_user_review_overlay.csv) |

Eight historical candidates/four sources; user visual decisions only, no quality pass or freeze. Crack470 mask retained1.0 with conflicting historical note retained; corrosion13 retained0.441285140562249, crop-only acceptance. Historical A0/A1 entries remain snapshots. Exact registration UTC is in the overlay; Exp13.2-B/SSL remain NO. No Batch02 or model execution.

## Exp13.2-A2 Batch 02 review preparation and Batch 01 audit archival (2026-10-23 UTC)

| Record | Status | Canonical evidence |
|---|---|---|
| Exp13.2-A2-Batch01-ChatGPTAudit | PASS / BOUNDED for registration and spatial audit only | [Audit archival](exp13_2_a2_batch01_chatgpt_audit.md); [original manifest archive](../results/exp13_restoration_feasibility/qa_20261023_batch01_manifest.csv) |
| Exp13.2-A2-Batch02-ReviewPacket | EXP13_2_A2_BATCH02_REVIEW_PACKET_PREPARED_AWAITING_CHATGPT_AUDIT | [Manifest](../results/exp13_restoration_feasibility/qa_20261023_batch02_manifest.csv); [user template](../results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_template.csv); [integrity](../results/exp13_restoration_feasibility/qa_20261023_batch02_integrity.json); [report](../results/exp13_restoration_feasibility/qa_20261023_batch02_review_report.md); [handoff](exp13_2_a2_batch02_handoff.md) |

Historical TRAIN R1 only: Crack86/260, corrosion2/8, each v1/v2;8variants/4independent sources. corrosion2 native mask retention0.5488721804511278, not repair success. Five review PNGs; all user decisions pending, quality/comment fields empty. Batch01 labels and historical records remain unchanged. No models, quality approval, freeze or training authorization; stop for ChatGPT/user review.

## Exp13.2-A2 Batch 02 user decision registration (2026-10-23T09:14:43.048190+00:00)

| Record | Status | Canonical evidence |
|---|---|---|
| Exp13.2-A2-Batch02-UserReview | EXP13_2_A2_BATCH02_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT | [Decision record](exp13_2_a2_batch02_human_review_decision.md); [overlay](../results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_overlay.csv); [validation](../results/exp13_restoration_feasibility/qa_20261023_batch02_registration_validation.json); [handoff](exp13_2_a2_batch02_user_review_handoff.md) |

8 user decisions /4 independent sources, with crop-only and reservation limits retained. Formal quality fields and original A0/Batch01/Batch02 artifacts unchanged. No pair freeze, quality PASS, Batch03 or training authorization. This appends a current registration state without replacing historical gates.

## Exp13.2-A2 cross-batch source eligibility audit (2026-10-23T09:34:10.618872+00:00)

| Record | Status | Evidence |
|---|---|---|
| Exp13.2-A2-CrossBatch-SourceEligibility | EXP13_2_A2_CROSS_BATCH_SOURCE_ELIGIBILITY_AUDITED_AWAITING_CHATGPT_AUDIT | [Audit and handoff](exp13_2_a2_cross_batch_source_eligibility_audit.md); [JSON](../results/exp13_restoration_feasibility/qa_20261023_cross_batch_source_eligibility.json) |

8source/16variant;7source with user acceptance,0with sufficient formal trusted-pair admission evidence. Proposed10/5/5 not frozen. Historical decisions retained; no formal quality PASS or further experimental authorization.

## Exp13.2-A2 Batch03 review preparation (2026-10-23T09:56:39.700061+00:00)

`BATCH03_REVIEW_PACKET_PREPARED_AWAITING_HUMAN_REVIEW`. [Handoff](exp13_2_a2_batch03_20261023T095257Z_handoff.md); [packet](../results/exp13_restoration_feasibility/batch03_20261023T095257Z_g3l3/README.md). Recomputed20sources minus8covered leaves12; select corrosion1/9/6 and Crack469 by status diversity then numeric ID,3corrosion+1Crack.8R1variants/4sources;13review PNGs; no models or new decisions. Historical overlays and A0 unchanged; all freeze/quality/Exp13.2-B/SSL remain NO. Stop for human/ChatGPT review.

## Exp13.2-A2 Batch03 user decision registration (2026-10-23T10:43:43.207444+00:00)

`EXP13_2_A2_BATCH03_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`. [Decision report](exp13_2_a2_batch03_human_review_decision.md); [overlay](../results/exp13_restoration_feasibility/qa_20261023_batch03_user_review_overlay.csv); [validation](../results/exp13_restoration_feasibility/qa_20261023_batch03_registration_validation.json); [handoff](exp13_2_a2_batch03_user_review_handoff.md). 8variants/4sources:3REJECT,2crop-scoped ACCEPT_WITH_SCOPE_LIMITATION,1ACCEPT,2blank pending. corrosion6 retains2789/4757 (0.5862938826991801),outside1968excluded; Crack469 versions independent; corrosion9 mechanical-hallucination feedback retained. Formal quality72cells blank; A0/Batch01/02 and original Batch03 unchanged. Freeze/quality/Exp13.2-B/SSL all NO. Stop for ChatGPT audit.

## Exp13.2-A2 Post-Batch03 source eligibility audit (2026-10-23T11:13:32.760984+00:00)

`EXP13_2_A2_POST_BATCH03_SOURCE_ELIGIBILITY_AUDITED_AWAITING_CHATGPT_REVIEW`. [Post-Batch03 audit](exp13_2_a2_post_batch03_source_eligibility_audit.md); [source ledger](../results/exp13_restoration_feasibility/qa_20261023_post_batch03_source_eligibility.json). Rebuilt from A0/manifests/three independent overlays: 20A0 sources; 12covered, 8uncovered; 5pending variant decisions; formal admission evidence sufficient=0. Visual acceptance, scope limitation, reservation and historical contradiction are separate strata. 10/5/5 remains an unconfirmed proposal. No freeze, quality PASS, training or Batch04 authorization; stop for ChatGPT review.

| Exp13 Phase 0 | AUDIT_COMPLETED_PENDING_CHATGPT_GATE | `docs/exp13_phase0_readonly_audit.md` and `results/exp13_phase0_readonly/20261024T093428Z/summary.json` |

| Exp13 Phase 0-R1 | AUDIT_COMPLETED_PENDING_CHATGPT_GATE | `docs/exp13_phase0_r1_audit_review.md` and `results/exp13_phase0_r1/20261024T100624Z/summary.json` |
| Exp13 Phase 1-A | COMPLETED_PENDING_CHATGPT_GATE | `docs/exp13_phase1a_handoff.md` and `results/exp13_phase1a/20261024T_PHASE1A_PROTOCOL/summary.json` |

Phase 1-A is a deterministic protocol/geometry audit only. Eight direct assertion tests pass; the frozen TRAIN pilot proposal and Burn source50 metadata are verified. No model download/load/inference, generated images, VAL/TEST access, training, SSL or trusted-pair authorization occurred. Runtime smoke remains separately gated.

## Exp13 Phase 1-A-R1 implementation repair (2026-10-24)

`EXP13_PHASE1A_R1_COMPLETED_PENDING_CHATGPT_GATE`. [Implementation review](exp13_phase1a_r1_implementation_review.md); [handoff](exp13_phase1a_r1_handoff.md); [evidence](../results/exp13_phase1a_r1/20261024T113936Z/summary.json).

The repair records real test execution with a standard-library fallback (9/9), compares all 1,266 TRAIN polygons against the historical Phase 0-R1 Pillow raster, validates native RGB multi-ROI full-image composition including Burn source50/all 7 polygons, and exercises persistent budget recovery and hard caps. No Diffusion/model call, VAL/TEST access, training, SSL, or raw-data write occurred. Smoke is a revised protocol only and remains unrun pending ChatGPT gate.

## Exp13 Phase 1-A-R2 minimal engineering closure (2026-10-24)

`EXP13_PHASE1A_R2_COMPLETED_PENDING_CHATGPT_GATE`. [Final engineering review](exp13_phase1a_r2_final_engineering_review.md); [handoff](exp13_phase1a_r2_handoff.md); [evidence](../results/exp13_phase1a_r2/20261024T123606Z/summary.json).

R2 adds non-identity RGB composition evidence, locked persistent budget accounting with cross-process contention tests, and corrected independent raster category statistics. Direct tests are 9/9 PASS; real TRAIN sources 2, 17, 50 and 13 pass native composition checks, including all seven Burn-50 polygons and five ROI groups. No model, Diffusion, generation, training, SSL, VAL/TEST access or raw-data write occurred; the result is pending ChatGPT gate.

## Exp13 Phase 1-B SDXL engineering Smoke preflight (2026-10-24)

`EXP13_PHASE1B_SMOKE_HOLD` / `PREFLIGHT_HOLD`. [Smoke review](exp13_phase1b_smoke_review.md); [handoff](exp13_phase1b_smoke_handoff.md); [evidence](../results/exp13_phase1b_smoke/20261024T130500Z/summary.json).

The required Hugging Face metadata request timed out before model download, so exact revision/license verification failed and no SDXL model was loaded or called. The R2 geometry candidates (source 2 single ROI, source 17 double ROI) remain unfrozen; no output or contact sheet exists. Pilot, fallback models, pair freeze and SSL remain prohibited.

## Exp13 Phase 1-B-R1 SDXL preflight recovery (2026-10-24)

`EXP13_PHASE1B_SMOKE_HOLD` / `NETWORK_HOLD`. [Review](exp13_phase1b_r1_review.md); [handoff](exp13_phase1b_r1_handoff.md); [evidence](../results/exp13_phase1b_r1/20261024T132018Z/summary.json).

AutoDL IPv4 TCP/HTTPS to Hugging Face failed and the Windows controlled host timed out, so no trusted official model acquisition path existed. Exact revision/license/file manifest remain unknown; no model artifact, transfer, load, inference or budget use occurred. Source 2/17 candidates remain unfrozen and Pilot/SSL remain prohibited.

## Exp13 Phase 1-B-R2 trusted model channel validation (2026-10-24)

`NETWORK_HOLD` / `MODEL_ARTIFACT_HOLD`. [Review](exp13_phase1b_r2_review.md); [handoff](exp13_phase1b_r2_handoff.md); [evidence](../results/exp13_phase1b_r2/20261024T134647Z/summary.json).

System DNS answers on Windows and AutoDL were inconsistent and could not be authenticated because Google/Cloudflare DoH were unreachable. Hugging Face metadata, fixed revision resolve and official LFS CDN were all unreachable. The supplied revision/license references remain unverified; no model artifact, transfer, load, inference, budget use, data access or configuration change occurred.

## Exp13 Phase 1-B-R3 offline SDXL artifact verification (2026-10-25)

`EXP13_PHASE1B_R3_ARTIFACT_VERIFIED_PENDING_CHATGPT_GATE`. [Review](exp13_phase1b_r3_artifact_review.md); [handoff](exp13_phase1b_r3_handoff.md); [evidence](../results/exp13_phase1b_r3/20261025T042354Z/summary.json).

All 19 server-side files match the supplied Windows manifest by size and SHA256; static FP16 structure and offline dependency checks pass. The manifest's official origin is supported by user-provided Windows snapshot/API evidence, not a server-side Hugging Face query. No model load, inference, budget use, VAL/TEST access, or training occurred; GPU Smoke remains unauthorized pending ChatGPT Gate.

## Exp13 Phase 1-B-S1 authorized GPU Smoke preflight (2026-10-25)

`EXP13_PHASE1B_S1_HOLD`. [Review](exp13_phase1b_s1_smoke_review.md); [handoff](exp13_phase1b_s1_handoff.md); [evidence](../results/exp13_phase1b_s1/20261025T044024Z/summary.json).

Stopped before model load because source 17 ROI1 reaches the image's right edge and overlaps unresolved FOV/frame/overlay content. Both authorized sources are required, so source 2 was not run alone. Local model hashes pass the supplied R3 manifest; GPU model load/inference and budget reservations remain zero. The input-only contact sheet is retained on AutoDL, not in Git.

## Exp13 Phase 1-B-S2 safe source reselection and SDXL Smoke (2026-10-25)

`EXP13_PHASE1B_S2_COMPLETED_PENDING_CHATGPT_GATE`. [Review](exp13_phase1b_s2_smoke_review.md); [handoff](exp13_phase1b_s2_handoff.md); [summary](../results/exp13_phase1b_s2/20261025T055350Z_smoke/summary.json); [ledger](../results/exp13_phase1b_s2/20261025T055350Z_smoke/call_ledger.json).

Three real TRAIN-only ROI calls ran for source 2 and dual-ROI source 65 using pinned offline SDXL. Native geometry, paste-back, and immutable outside-support checks passed; 3/4 Smoke calls are occupied. Semantic quality failed: source 2 has a black rectangular artifact and source 65's small ROI is dark/implausible. Neither is an approved pseudo-normal sample. The candidate audit's null selection and malformed process exit-code capture remain explicit audit limitations. Stop for ChatGPT review; no Pilot or training is authorized.

## Exp13 Phase 1-B-S3 zero-inference artifact root-cause audit (2026-10-25)

| Record | Status | Evidence |
|---|---|---|
| Exp13 Phase 1-B-S3 | EXP13_PHASE1B_S3_DIAGNOSIS_COMPLETED_PENDING_CHATGPT_GATE | [Root-cause review](exp13_phase1b_s3_root_cause_review.md); [handoff](exp13_phase1b_s3_handoff.md); `results/exp13_phase1b_s3/20261025T141700Z/` |

CPU-only comparisons of existing S2 source 2/65 Canvas, mask, inverse and paste stages localize the source-2 bar and source-65 roi_1 darkening to raw SDXL Canvas output. Manual review additionally records UI/FOV/structure contamination and reflect-padding risks. Mask padding leakage is zero, outside-support changes remain zero, and the S2 ledger stays at 3/4. The final Smoke slot is held; no output is an approved pseudo-normal pair.

## Exp13 Phase 1-B-S4 restoration eligibility feasibility (2026-10-25)

| Record | Status | Evidence |
|---|---|---|
| Exp13 Phase 1-B-S4 | RESTORATION_ELIGIBILITY_PREPARED_AWAITING_HUMAN_REVIEW | [Eligibility review](exp13_phase1b_s4_eligibility_review.md); [handoff](exp13_phase1b_s4_handoff.md); `results/exp13_phase1b_s4/20261025T075800Z/` |

TRAIN-only CPU screening found 1,266 polygons/668 images, with 816 geometry-proxy polygons from 530 sources. Twenty contact sheets cover all seven labels and 20 distinct sources, but no candidate is restoration-eligible before full-context human review. At `scale<=4x` and Canvas content `>=75%`, 26 polygons/26 sources remain. S2 connected-component precision was added without rewriting S3. No model load, inference, fourth Smoke call, VAL/TEST access, raw write, Pilot or training occurred.

## Exp13 Phase 1-B-S4-R1 strict candidate review preparation (2026-10-25)

`EXP13_PHASE1B_S4_R1_STRICT26_REVIEW_PREPARED_PENDING_CHATGPT_GATE`. [Review preparation](exp13_phase1b_s4_r1_strict_review_preparation.md); [handoff](exp13_phase1b_s4_r1_handoff.md); evidence `results/exp13_phase1b_s4_r1/20261025T171500Z_s4r1/`.

The strict predicate was independently reapplied and geometry hashes rebuilt: 26 polygons / 26 independent TRAIN sources (Material missing 14, Burn 8, corrosion 4), with zero overlap against the original 20-sheet packet. AutoDL contains 26 CPU-only review sheets, a category overview, pending template and a 16,536,565-byte ZIP (SHA256 `a29af7cd862ad3fe433dcaf2ea068216d95ca909208848a2af13df51a1f2bd3f`). No candidate is restoration-eligible; S2 budget remains 3/4 and no model or training action occurred.

## Exp13 Phase 1-B-S4-R2 visual review archival and candidate independence audit (2026-10-25)

`EXP13_PHASE1B_S4_R2_COMPLETED_PENDING_CHATGPT_GATE`. [Review](exp13_phase1b_s4_r2_review.md); [handoff](exp13_phase1b_s4_r2_handoff.md); evidence `results/exp13_phase1b_s4_r2/20261025T092025Z/`.

The 26 S4-R1 review images are archived as `CHATGPT_PRELIMINARY_VISUAL_REVIEW`: 24 are not recommended for the current compact ROI text-only SDXL protocol, while 285/378 remain method-design review only; formal restoration eligibility and trusted pairs remain zero. CPU/TRAIN-only source audit verified 26 unique source/image/annotation hashes, no exact collisions and 325 pairwise records; exploratory near-duplicate proxy pairs were zero, but physical scene independence is unproven. Current SDXL is held, DreamShaper is audit-only, and classical CPU texture repair is only a separately authorized control. S2 remains 3/4 with one Smoke call unused; no model, VAL/TEST, raw write, Pilot, SSL or training action occurred.

## Exp13.2-A3 DreamShaper historical restoration review preparation (2026-10-25)

`EXP13_2_A3_DREAMSHAPER_REVIEW_PREPARED_PENDING_CHATGPT_GATE`. [Review](exp13_2_a3_dreamshaper_quality_review_preparation.md); [handoff](exp13_2_a3_handoff.md); evidence `results/exp13_2_a3_dreamshaper_review/20261025T203500Z_a3/`.

TRAIN-only CPU audit indexes all 88 historical candidates (40 R1, 48 R2 diagnostics) from 20 source groups and crosswalks 16 prior user decisions from eight R1 groups. The eight-source quality review ZIP is controlled on AutoDL (14,554,299 bytes; SHA256 `cb0160dba7bc957ff8ae8c48615c381fa2898febc32337b049db335bba895565`); images/ZIP are excluded from Git. Current quality fields remain pending, formal passes/trusted pairs are zero, ten sources retain crop truncation risk, and outside-support changes remain a review concern. S2 budget is unchanged at 3/4; no model/inference, VAL/TEST, raw writes, or training occurred. Audit process HEAD `787a3be0190dfa9b11c5416a5b686a6bae4598d9` differs from reporting branch start `35a2405d549ccdc21ea06eff0a8e91b9face1749` and is documented in provenance.

## Exp13.2-A4 DreamShaper targeted restoration quality evidence (2026-10-25)

`EXP13_2_A4_TARGETED_QUALITY_EVIDENCE_PREPARED_PENDING_CHATGPT_GATE`. [Review](exp13_2_a4_targeted_quality_review.md); [handoff](exp13_2_a4_handoff.md); evidence `results/exp13_2_a4_targeted_quality/20261025T221500Z_a4/`.

CPU-only evidence covers five shortlist variants from three source groups; no semantic PASS or trusted pair is created. The source-2 DreamShaper/SDXL comparison is labeled `CROSS_PROTOCOL_CASE_STUDY_NOT_MODEL_RANKING`; actual source hashes differ and the persisted S2 mask/native dimensions are not directly comparable. The 16/16 corrected decision crosswalk preserves historical labels. S2 remains 3/4 with one slot held; no model calls, training, VAL/TEST access or raw writes occurred. Industrial review PNGs/ZIP remain controlled on AutoDL and are excluded from Git.

## Exp13 Phase 1-B-S5 matched-input SDXL single-call study (2026-10-25)

`EXP13_PHASE1B_S5_COMPLETED_PENDING_CHATGPT_GATE`. [Review](exp13_phase1b_s5_matched_input_review.md); [handoff](exp13_phase1b_s5_handoff.md); evidence `results/exp13_phase1b_s5/20261025T230000Z_s5/`.

One TRAIN source (`Crack:260`) used the historical DreamShaper center crop and reconstructed mask as a fixed 1024 SDXL input, with no reflect padding. The one authorized call succeeded under the inherited successor ledger (4/4 occupied, 0 remaining). The raw aligned output has 66,735 >5 changes outside the mask; semantic quality and training-pair admission remain pending/zero. This is a bounded input-protocol study and not model ranking; controlled review images remain on AutoDL and are excluded from Git.

## Exp13 cloud image restoration G1B–G1F (historical registration, 2026-11-01)

[Unified cloud index](exp13_cloud/EXP13_CLOUD_INDEX.md) records all eight stages, code entries, fixed prompts, private asset IDs/hashes and historical POST counts. [Sanitized registry](exp13_cloud/experiment_registry.csv) and [handoff](exp13_cloud/GITHUB_AUDIT_HANDOFF.md) separate preparation from actual generation.

G1B: one raw corrosion23 result; G1C/G1D: `FAIL_PAIR_FIDELITY / HOLD_EXPANSION`; G1E-0: prepared; G1E-1: two successful synthetic calls with `MASK_CONDITIONED_SPATIAL_RESPONSE_OBSERVED`; G1F-0/0B: corrosion2 ROI preparation, B4:3 selected; G1F-1A: `PREFLIGHT_PASS`; G1F-1: `LIVE_NOT_STARTED`. No trusted Pair, batch or training authorization follows. The Git sync adds zero API calls or scientific runs and preserves Exp11/Exp12.

## Exp13-G1F-1 single cloud repair saved-result audit (2026-11-01)

`G1F_1_RAW_AUDIT_PREPARED / PENDING_CHATGPT_VISUAL_GATE`. [Result](exp13_cloud/reports/G1F_1_summary.md); [updated cloud index](exp13_cloud/EXP13_CLOUD_INDEX.md); [handoff](exp13_cloud/GITHUB_AUDIT_HANDOFF.md). The user subsequently executed one authorized corrosion2 call, HTTP 200; the raw output is 1024x1024 PNG/RGB versus the 320x240 input. One occupied terminal budget, no retry. The earlier LIVE_NOT_STARTED text above is the preserved preflight/sync snapshot. Cloud historical generation total is now four; the present CPU audit/sync adds zero. Geometry/appearance diagnostics and private review sheets prepare all four pending visual Gates, without Pair/G2/training authority.

## Exp13-G1F-2 fixed offline geometric correspondence (2026-11-01)

`GEOMETRY_CANDIDATE` / `G1F_2_ALIGNMENT_AUDIT_COMPLETE / PENDING_CHATGPT_AND_USER_PAIR_GATE`. [Report](exp13_cloud/reports/G1F_2_summary.md), [source/protocol](../tools/exp13_cloud/g1f2/). Native SIFT/Lowe .75, 3px RANSAC, full/UI-excluded comparisons and two fixed linear models. Both non-UI models have 85/85 dispersed inliers with <0.26px median error; valid masks and physical-edge/cavity views are private. A documented JSON-only bug correction used a new directory with identical scientific parameters; no search. User appearance acceptance does not grant strict pixel Pair or training eligibility. Added API POST/GPU/VAL/TEST/training=0; no G2.

## Exp13 G2 three-source cloud Pilot preparation (2026-11-02)

`G2_THREE_SOURCE_PREPARATION_COMPLETE / PENDING_CHATGPT_GATE`. [Sanitized frozen manifest](exp13_cloud/g2/G2_THREE_SOURCE_MANIFEST.json); [preparation report](exp13_cloud/reports/G2_three_source_preparation.md); [offline preparation code](../tools/exp13_cloud/g2/prepare_three_source_pilot.py).

Frozen TRAIN sources are Crack260 (`singleton_260`), Crack470 (`near_0007`) and corrosion23 (`singleton_23`), with full-frame native ROIs and professional original JSON Polygons. Pilot-4 ZIP SHA256 `d9f049bd…04f18` and frozen Split Manifest SHA256 `35d577c1…9634d` matched. corrosion23's G1B/G1C/G1D history is disclosed but does not count as a G2 result; G1F corrosion2 is excluded from the cohort and denominator. The exploratory Gate requires records for all three sources and at least one credible candidate per class; it makes no significance claim and does not authorize training. The ceiling is three first-call POSTs total, one per source; this stage used 0 and authorizes none. Private industrial assets remain local.

## Exp13 G2 evaluation revision 2 and single-source executors (2026-11-02)

`G2_PROTOCOL_CORRECTED / THREE_SOURCE_EXECUTOR_PREPARED / PENDING_CHATGPT_CODE_GATE_AND_USER_PAID_AUTHORIZATION`. [Evaluation revision](exp13_cloud/g2/G2_EVALUATION_REVISION_2.json); [executor preparation report](exp13_cloud/reports/G2_protocol_executor_preparation.md); [frozen result audit protocol](exp13_cloud/g2/G2_RESULT_AUDIT_PROTOCOL.md).

Only Crack260 and Crack470 are new-source outcomes (`NEW_CLOUD_SOURCE`, n=2). corrosion23 is a historical failure targeted retest, reported separately (n=1); its G1B failure is not counted again. G1F corrosion2 is excluded. All three newly authorized calls, if later approved, have separate request/usage/cost records. The mixed-class exploratory Gate requires all three outcomes, at least one credible Crack candidate, and a credible corrosion23 retest. It does not claim significance or authorize training.

Three fixed wrappers were statically checked and each completed an offline multipart and CurrentUser DPAPI Dry Run against final execution-bundle SHA256 `cee4f8efacab9027dcbcfa8b97e17d1edfb5ed72947bb3fc459fcd317ddfa115`. Each used the same frozen prompt file hash and exact transmitted-string hash. Requests were not sent; API POST/network=0, and no approval receipt or production STARTED lock exists. Future commands are recorded in the report and remain gated by source-specific receipt and user confirmation.

## Exp13-G3-B2-A offline batch preparation (2026-11-07)

`G3B2A_DRY_RUN_READY_FOR_REVIEW`. [Sanitized review](exp13_cloud/G3_B2_A_BATCH_PREPARATION_REVIEW.md) reports 523 SELECT sources / 901 targets, 356 metadata-only routine-shape candidates, 167 special-review sources, and zero confirmed safe sources while UI/mechanical-edge evidence remains unknown. A private 30-source proposal (6 each Burn/Crack/Dent/Material missing/Tears) covers 31 targets and is group-deduplicated; all rows remain PROPOSED with no paid authorization. Native-size selected-only masks and CPU overlays passed the dry-run. Prompt hashes are pinned, historical G2 protocol bytes are checked against its frozen bundle, provider constraints and cost remain unknown, and geometry/Pair admission remain separate. Tests 4/4 PASS; API POST/GPU/TEST/AutoDL/thesis=0. Stop for independent review and separate authorization.

## Exp13-G2 closeout and G3-A preparation (2026-11-07)

`G3A_WAITING_AUTODL_TRAIN_EXPORT`. The three saved G2 calls were audited once
with fixed CPU evidence checks and G1F-2 geometry rules. Execution is verified
at one POST/HTTP 200 per source; cost fields remain `NOT_RETURNED`. User visual
acceptance is separate from Pair usability. Crack260 is `GEOMETRY_CANDIDATE`;
Crack470 and corrosion23 are `GEOMETRY_UNVERIFIED`; all Pair/training states
remain pending/false. Windows lacks the complete 668-image TRAIN image/JSON
export, so the full private atlas is waiting for AutoDL no-GPU export. See
[G2 closeout](exp13_cloud/reports/G2_closeout_20261107.md),
[sanitized ledger](exp13_cloud/g2/G2_CLOSEOUT_LEDGER.json),
[G3-A handoff](exp13_cloud/G3_A_PREPARATION_HANDOFF.md),
[G3-B reuse plan](exp13_cloud/G3_B_REUSE_PLAN.md), and the new tools under
`tools/exp13_cloud/g3/`. API POST/GPU training/TEST image access remain 0 for
this local work.
