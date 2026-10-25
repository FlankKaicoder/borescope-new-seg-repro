# Experiment index

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
