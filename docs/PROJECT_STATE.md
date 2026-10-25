# Project state

| Field | Final fact |
|---|---|
| Current phase | `PROJECT_COMPLETE` |
| Option decision | `OPTION_A_DIRECT_FINALIZATION` / `NO_MORE_MODEL_EXPERIMENTS` |
| Dataset | `/root/autodl-tmp/borescope-new-seg-data/v1`; 969 images; 1847 instances; 7 classes |
| Split | train 668/1266, val 154/296, test 147/285; SHA256 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Candidate Freeze | `PASS`; commit `9991fcfcb9cf6c0ab8920ad7deadeed579ce5585`; completed before TEST access |
| Final method | `YOLO11n-seg Baseline`; seed44 |
| Final checkpoint | `/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt` |
| Checkpoint SHA256 | `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9` |
| Frozen VAL Mask mAP50-95 | `0.3251567516159603` |
| Final TEST Mask mAP50-95 | `0.2716207089319057` |
| TEST access | `true`; phase `ONE_FINAL_FROZEN_EVALUATION` |
| Exp11 execution note | Initial premetric run interrupted and preserved; exactly one unchanged retry explicitly authorized and completed |
| Model selection closed | `true` |
| Post-TEST training | `NO` |
| Post-TEST threshold/model/seed selection | `NO` |
| Hard Mining | `HARD_MINING_NOT_CONFIRMED` |
| Resolution | `NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE / FUTURE_WORK_ONLY` |
| Formal training allowed | `NO` |
| Next allowed experiment | `NONE` |
| Exp11 results commit | `fb584621b816de8f344d49daeba2656caee46e92` |
| Final docs commit | `b094d16bacb325e44d61eb4bedad54b00d231790` |

No required experiment remains. Future work is outside the closed experimental phase.

## Post-project independent extension

- `Exp12.0` SimSiam encoder/audit design was added on 2026-08-21 as an independent post-project study.
- Status: `EXP12_3_PASS_SINGLE_SEED_FROZEN_VAL_POSITIVE`.
- It does not reopen model selection, modify Exp00–Exp11, replace the final Baseline, or alter the frozen TEST conclusion.
- Exp12.0 runtime probe PASS on instance `b67c4c8bad-03c084fc`: TRAIN exact-set 668, encoder layers 0–10, output `[2,256,16,16]`, 120 parameter tensors, 1,365,472 parameter elements and 240 state tensors.
- Framework load had `requires_grad=True` for `0/120` encoder parameters. No optimizer, backward, training, VAL/TEST image access or checkpoint occurred.
- Exp12.1-S PASS on 2026-08-22: explicit unfreeze changed `requires_grad=True` from 0/120 to 120/120; optimizer identity covered 120/120 exactly once with no duplicates.
- The first real TRAIN batch produced present/finite/non-zero gradients for 120/120 encoder parameters and changed 120/120 parameters. After one epoch, `changed_ratio=1.0`; checkpoint and export reload matched 120/120 parameter hashes; VAL/TEST image access remained 0/0.
- Exp12.1 formal TRAIN-only SimSiam completed 100/100 epochs with all metrics finite and no collapse threshold breach; fixed epoch 100 was used without early stopping or best-checkpoint selection.
- Exp12.2 PASS: 120/120 encoder parameter tensors and 1,365,471/1,365,472 parameter elements changed; final checkpoint and backbone export matched in-memory parameter hashes 120/120.
- Formal SSL used TRAIN exact-set 668; VAL/TEST image access remained 0/0.
- Exp12.3-S and formal Route A/B fairness Gates PASS: Route B changed 120/120 backbone parameters, survived Trainer construction exactly, and A/B neck/head plus first-batch hashes matched.
- Both routes completed 100/100 epochs, 66,800 TRAIN draws and 1,066 optimizer steps with finite TRAIN losses and reloadable checkpoints; TEST access was zero.
- Frozen VAL Mask mAP50-95 was 0.298981 for Route A and 0.321897 for Route B, delta +0.022916. Mask precision increased while mask recall decreased.
- This is a positive single-seed VAL signal only; it does not replace the final Baseline or modify Exp00–Exp11. Multi-scale work remains outside scope pending separate authorization.

### Authorized Exp12 continuation (2026-08-23)

- This section supersedes the earlier `Multi-scale work remains outside scope pending separate authorization` boundary: the user explicitly authorized completing the remaining Exp12 work.
- Current extension status: `EXP12_COMPLETE_LOCAL_CHANGE_VALID_PROXY_NO_FROZEN_VAL_MASK_GAIN`.
- Exp12.4-S and 30-epoch formal multi-scale local-change SSL PASS; first-step and formal final audits changed 120/120 backbone tensors, and formal changed all 1,365,472 elements.
- Exp12.4 formal proxy IoU ended at 0.952284 without collapse; checkpoint/export round trip matched 120/120 parameter hashes; SSL VAL/TEST access stayed 0/0.
- Exp12.5 Route C fairness and Trainer-initialization Gates PASS; 100 epochs, 66,800 TRAIN draws and 1,066 optimizer steps completed; TEST access was zero.
- Frozen VAL Mask mAP50-95: Route A COCO 0.298981, Route B SimSiam 0.321897, Route C local change 0.293530. Route C minus A = -0.005451; Route C minus B = -0.028367.
- The local-change proxy is valid but produced no single-seed frozen-VAL Mask mAP50-95 gain. Exp12 is complete and Exp00-Exp11 remain unchanged.


## Post-project Exp13 extension

- Exp13.1-R2 controlled generation diagnosis completed on 2026-09-21 with four fixed TRAIN-only sources and 48 successful outputs. It is waiting for visual review and does not reopen the closed project conclusions. No VAL/TEST access, SSL, YOLO training, or downstream training occurred.

### Exp13.1-R2-QA final quality archival (2026-10-21)

- Status: `EXP13_1_R2_QUALITY_GATE_ARCHIVED`; documentation archival only.
- Final quality Gate: `GENERIC_TEXT_ONLY_INPAINTING_NOT_APPROVED_FOR_CURRENT_SSL_SYNTHESIS`.
- This dated decision follows the waiting-review snapshot above; historical R1/R2 Gates and runs are retained. R1 completed 40/40 outputs but failed qualitative quality review; R2 completed 48/48 outputs from four TRAIN-only sources.
- Evidence is the user's authorized account of earlier ChatGPT qualitative review of uploaded contact/diagnosis sheets, not a new review, expert annotation, or complete quantitative scoring. Existing blank review scores remain blank.
- Scope: current DreamShaper-8-Inpainting checkpoint, Crack/corrosion task, and current generation/controlled-diagnosis protocol; no claim that all Diffusion models fail.
- `CURRENT_TEXT_ONLY_INPAINTING_SSL_AUTHORIZATION = NO`
- `SEVEN_CLASS_GENERATION_AUTHORIZATION = NO`
- `FURTHER_R3_PARAMETER_SWEEP = NO`
- `EXP13_2_GENERATION_AUTHORIZATION = NO`
- Ready for Exp13.2-A method research/design discussion only; generation requires separate authorization. Exp00–Exp11 and Exp12 conclusions remain unchanged.
- Evidence: [final quality review](exp13_1_r2_final_quality_gate_review.md); [QA handoff](exp13_1_r2_qa_archival_handoff.md).

### Exp13.2-A0 historical restoration review preparation (2026-10-21)

- Status: `RESTORATION_REVIEW_ASSETS_PREPARED_AWAITING_HUMAN_REVIEW`; user-authorized historical asset preparation only.
- R1 40 outputs/20 sources + R2 48 outputs/4 overlapping sources = 88 candidates/20 source groups (Crack/corrosion 10 sources each). Variants are correlated, not independent samples.
- All referenced source/mask/output files decoded; historical central crop/fit to 512x512 reproduced. Ten source masks are partly cropped (minimum retained area 44.1285%); cropped-away defects cannot count as restoration.
- Prepared 20 source review sheets, two class overviews, diagnostic differences, spatial/file reports and an 88-row human review CSV. All 792 human judgment cells remain blank; no candidate accepted or scored.
- All 195 historical R1/R2 files retained identical SHA256; Phase C1 untracked files, datasets, model caches and previous Gates/templates preserved. No generation/training or VAL/TEST image access.
- Annotation-derived polygon masks must be disclosed as supervision in any future design. Credible pseudo-normal pair count and source coverage remain unknown.
- SSL, seven-class generation, R3 sweep, Exp13.2 generation and Exp13.2-B authorizations remain NO. Stop for ChatGPT/user visual review.
- Evidence: [A0 preparation report](exp13_2_a0_restoration_feasibility_preparation.md); [A0 handoff](exp13_2_a0_restoration_feasibility_handoff.md); `results/exp13_restoration_feasibility/a0_20261021_081812/`.

### Exp13.2-A0-QA source visual screening archival (2026-10-21)

- Status: `EXP13_2_A0_SOURCE_VISUAL_SCREENING_COMPLETE`; source-level qualitative first-pass screening archived only.
- Screening of 20 source sheets: 15 promising for further review, 2 uncertain, 3 corrosion sources restricted. This is not 15 accepted training pairs or a pass-rate estimate.
- Candidate ledger covers all 88 historical variants. Crack 111 v1/v2 have qualitative notes only and still require variant review; remaining 86 are pending. No variant PASS.
- Original A0 blank quality template remains unchanged; 792 judgment cells blank. Formal candidate-level review incomplete; trusted repair-pair set not frozen.
- Exp13.2-A1 method discussion may continue. `FINAL_TRUSTED_PAIR_FREEZE = NO`, `RESTORATION_QUALITY_GATE_PASS = NO`, `EXP13_2_B_AUTHORIZATION = NO`, `SSL_AUTHORIZATION = NO`.
- Evidence source: ChatGPT qualitative screen of user-uploaded source-level review sheets, not industrial-expert review, full variant scoring, or statistical significance. See [screening report](exp13_2_a0_source_visual_screening_review.md), [handoff](exp13_2_a0_source_visual_screening_handoff.md), and `results/exp13_restoration_feasibility/qa_20261021_090000/`.

## Exp13.2-A1 method design (2026-10-21)

Status: `EXP13_2_A1_DESIGN_READY_FOR_CHATGPT_REVIEW`. This is documentation-only; the design is not frozen and no implementation, data construction, training, inference, generation, or pair selection is authorized. The source-level A0 screening is not candidate-level quality approval. `FINAL_TRUSTED_PAIR_FREEZE = NO`, `RESTORATION_QUALITY_GATE_PASS = NO`, `EXP13_2_B_AUTHORIZATION = NO`, and `SSL_AUTHORIZATION = NO`. See [A1 handoff](exp13_2_a1_design_handoff.md).

## Exp13.2-A1-R1 consistency repair (2026-10-23)

Status: `EXP13_2_A1_R1_DESIGN_READY_FOR_CHATGPT_REVIEW`. The user-relayed A1 review was conditional and requested specification repair. P/N/I localization regions, identical T2/T3 difference heads, proposed budgets and future artifact gates are clarified in [R1 response](exp13_2_a1_r1_design_review_response.md) and [R1 handoff](exp13_2_a1_r1_handoff.md). Historical statuses above remain snapshots. Formally trusted/frozen pairs remain 0; the proposed exploratory minimum is at least 10 independent trusted sources, Crack/corrosion each at least 5, pending approval and not statistical sufficiency.

`EXP13_2_A1_DESIGN_FROZEN = NO`; `FINAL_TRUSTED_PAIR_FREEZE = NO`; `RESTORATION_QUALITY_GATE_PASS = NO`; `EXP13_2_B_AUTHORIZATION = NO`; `SSL_AUTHORIZATION = NO`. Design approval is not Data Quality PASS, data approval is not training authorization, and smoke PASS is not formal SSL approval. No implementation, control-data generation, inference, training, or VAL/TEST image access.

## Exp13.2-A2 Batch 01 user decisions and spatial audit (2026-10-23)

Status: `EXP13_2_A2_BATCH01_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`. User-supplied decisions for 8 historical candidates / 4 independent sources are registered in a separate [overlay](../results/exp13_restoration_feasibility/qa_20261023_batch01_user_review_overlay.csv); no historical screen, template or Gate was overwritten. Three decisions remain pending; Crack111 v2 is visually accepted, Crack470 v1/v2 retain the user's visual-accepted/pending-spatial status without ranking, and corrosion13 v1/v2 are accepted only within the historical crop. Crack146 green marking/blur originated in the input per user and cannot directly be called generation artifacts.

Read-only native-mask audit: Crack470 retains3134/3134 pixels (1.0), despite the preserved conflicting historical crop warning; corrosion13 retains2747/6225 (0.441285140562249) and excludes3478 annotated pixels. Spatial grid consistency is not semantic repair certification. Exact UTC/provenance, counts and hashes: [decision report](exp13_2_a2_batch01_human_review_decision.md), [spatial report](exp13_2_a2_batch01_spatial_audit.md), [handoff](exp13_2_a2_batch01_handoff.md). Historical A0 template remains88rows/792blank cells. No new visual review or model execution.

`FINAL_TRUSTED_PAIR_FREEZE = NO`; `RESTORATION_QUALITY_GATE_PASS = NO`; `EXP13_2_B_AUTHORIZATION = NO`; `SSL_AUTHORIZATION = NO`. A1 design remains unfrozen. Stop for ChatGPT audit; no Batch02, generation, training or subsequent-stage execution.

## Exp13.2-A2 Batch 02 review packet and Batch 01 audit archival (2026-10-23 UTC)

Status: `EXP13_2_A2_BATCH02_REVIEW_PACKET_PREPARED_AWAITING_CHATGPT_AUDIT`. The user-relayed Batch 01 registration/spatial audit is archived as `PASS / BOUNDED` in [audit record](exp13_2_a2_batch01_chatgpt_audit.md). Its three pending user decisions and original overlay remain unchanged. Crack470's registration-time pending-spatial label is preserved separately from the completed technical audit (3134/3134 pixels retained); historical warning remains. corrosion13 acceptance remains crop-only (2747/6225 retained). The original Batch01 manifest is now archived byte-identically with its expected hash.

Prepared only historical TRAIN R1 Crack86/260 and corrosion2/8 v1/v2: 8 candidates / 4 independent sources, 0 join failures, 5 review PNGs. All file/dimension/crop checks agree; corrosion2 mask retained1314/2394 (0.5488721804511278), so cropped-away annotation cannot count as repaired. Other three source fractions are1.0. All user decisions pending; quality/comment fields blank; technical audit and visual decisions are separate. Original A0 template stays88rows/792blank cells. No new quality judgment or models. See [review report](../results/exp13_restoration_feasibility/qa_20261023_batch02_review_report.md), [manifest](../results/exp13_restoration_feasibility/qa_20261023_batch02_manifest.csv) and [handoff](exp13_2_a2_batch02_handoff.md).

`FINAL_TRUSTED_PAIR_FREEZE = NO`; `RESTORATION_QUALITY_GATE_PASS = NO`; `EXP13_2_B_AUTHORIZATION = NO`; `SSL_AUTHORIZATION = NO`. Historical entries remain snapshots; this authorization permits review materials only. Stop after Git delivery for ChatGPT audit and user visual review; no generation, inference, training or subsequent-stage execution.

## Exp13.2-A2 Batch 02 subsequent user decision registration (2026-10-23T09:14:43.048190+00:00)

`EXP13_2_A2_BATCH02_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`. [Decision report](exp13_2_a2_batch02_human_review_decision.md), [overlay](../results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_overlay.csv), [handoff](exp13_2_a2_batch02_user_review_handoff.md). Exactly 8 user decisions / 4 source groups: 4 ACCEPT, 2 crop-scoped ACCEPT_WITH_SCOPE_LIMITATION, 1 REJECT, 1 ACCEPT_WITH_RESERVATION. These are user visual decisions only, not quality PASS or frozen pairs. corrosion2 retains 1314/2394 annotation pixels; outside1080 remain outside acceptance. corrosion8 v1/v2 decisions stay separate. Original A0/Batch01/Batch02 preparation records are unchanged historical snapshots. Freeze/quality/Exp13.2-B/SSL remain NO. No subsequent experiment; await ChatGPT audit.

## Exp13.2-A2 cross-batch source eligibility audit (2026-10-23T09:34:10.618872+00:00)

`EXP13_2_A2_CROSS_BATCH_SOURCE_ELIGIBILITY_AUDITED_AWAITING_CHATGPT_AUDIT`. [Audit/handoff](exp13_2_a2_cross_batch_source_eligibility_audit.md), [JSON](../results/exp13_restoration_feasibility/qa_20261023_cross_batch_source_eligibility.json). Deduplicate16variants to8sources (Crack5/corrosion3);7sources have user acceptance (4/3), but0have sufficient documented formal admission evidence. Exclusive strata:3 ordinary candidates,1 historical-note-conflict acceptance,2 crop-scoped acceptances,1 reservation acceptance,1 pending-only source. Proposed10/5/5 threshold unfrozen; formal-evidence deficit10/5/5. Batch01 three pending variants and both overlays unchanged. Two Batch02 wording corrections clarify normalized summaries are not verbatim quotes. Freeze/quality/Exp13.2-B/SSL remain NO; no model or next-stage execution.

## Exp13.2-A2 Batch03 review preparation (2026-10-23T09:56:39.699942+00:00)

`BATCH03_REVIEW_PACKET_PREPARED_AWAITING_HUMAN_REVIEW`. [Handoff](exp13_2_a2_batch03_20261023T095257Z_handoff.md); [packet](../results/exp13_restoration_feasibility/batch03_20261023T095257Z_g3l3/README.md). Recomputed20sources minus8covered leaves12; select corrosion1/9/6 and Crack469 by status diversity then numeric ID,3corrosion+1Crack.8R1variants/4sources;13review PNGs; no models or new decisions. Historical overlays and A0 unchanged; all freeze/quality/Exp13.2-B/SSL remain NO. Stop for human/ChatGPT review.

## Exp13.2-A2 Batch03 user decision registration (2026-10-23T10:43:43.207444+00:00)

`EXP13_2_A2_BATCH03_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`. [Decision report](exp13_2_a2_batch03_human_review_decision.md); [overlay](../results/exp13_restoration_feasibility/qa_20261023_batch03_user_review_overlay.csv); [validation](../results/exp13_restoration_feasibility/qa_20261023_batch03_registration_validation.json); [handoff](exp13_2_a2_batch03_user_review_handoff.md). 8variants/4sources:3REJECT,2crop-scoped ACCEPT_WITH_SCOPE_LIMITATION,1ACCEPT,2blank pending. corrosion6 retains2789/4757 (0.5862938826991801),outside1968excluded; Crack469 versions independent; corrosion9 mechanical-hallucination feedback retained. Formal quality72cells blank; A0/Batch01/02 and original Batch03 unchanged. Freeze/quality/Exp13.2-B/SSL all NO. Stop for ChatGPT audit.

## Exp13.2-A2 Post-Batch03 source eligibility audit (2026-10-23T11:13:32.760984+00:00)

`EXP13_2_A2_POST_BATCH03_SOURCE_ELIGIBILITY_AUDITED_AWAITING_CHATGPT_REVIEW`. [Post-Batch03 audit](exp13_2_a2_post_batch03_source_eligibility_audit.md); [source ledger](../results/exp13_restoration_feasibility/qa_20261023_post_batch03_source_eligibility.json). Rebuilt from A0/manifests/three independent overlays: 20A0 sources; 12covered, 8uncovered; 5pending variant decisions; formal admission evidence sufficient=0. Visual acceptance, scope limitation, reservation and historical contradiction are separate strata. 10/5/5 remains an unconfirmed proposal. No freeze, quality PASS, training or Batch04 authorization; stop for ChatGPT review.

### Exp13 Phase 0 read-only data and ROI audit (2026-10-24)

- Status: `EXP13_PHASE0_AUDIT_COMPLETED_PENDING_CHATGPT_GATE`.
- Frozen TRAIN-only audit completed on 668 images and 1,266 valid polygon instances; all seven classes were covered and 844 native-coordinate ROIs were planned.
- Gates: 0-A PASS, 0-B CONDITIONAL_PASS (geometry usable; gray/padding/FOV border remains heuristic), 0-C PASS, 0-D PASS, 0-E PASS.
- No VAL/TEST access, model download/inference, diffusion, synthetic generation, training, SSL, or raw-data write occurred. No trusted pair or Exp13.2-B/SSL authorization is implied.
- Evidence: `results/exp13_phase0_readonly/20261024T093428Z/`; audit and handoff: `docs/exp13_phase0_readonly_audit.md` and `docs/exp13_phase0_readonly_handoff.md`.

### Exp13 Phase 0-R1 budget and geometry audit repair (2026-10-24)

- Status: `EXP13_PHASE0_R1_AUDIT_COMPLETED_PENDING_CHATGPT_GATE`; R0 outputs remain unchanged.
- Frozen TRAIN-only recheck: 668 images, 1,266 polygons, original 844 ROI merge reproduced; exact polygon and source-union coverage ratio 1.0.
- R1 budget: revised 21-source proposal contains 27 ROIs and 54 planned pilot calls; actual model calls 0. Smoke 4, pilot/revision 74, mean 100-source total 284, p90 total 400; budget is `CONDITIONAL_FEASIBLE`.
- Boundary-risk recomputation is 317/1266 (25.04%) across 202 sources under the documented geometric rule; gray/black/padding remains `UNRESOLVED_HEURISTIC`; merge/re-paste risks remain disclosed.
- Gates: R1-A CONDITIONAL_PASS, R1-B PASS, R1-C CONDITIONAL_PASS, R1-D CONDITIONAL_PASS, R1-E PASS. No Diffusion, training, SSL, trusted-pair freeze, or Exp13.2-B authorization.
- Evidence: `results/exp13_phase0_r1/20261024T100624Z/`; review and handoff: `docs/exp13_phase0_r1_audit_review.md` and `docs/exp13_phase0_r1_handoff.md`.
### Exp13 Phase 1-A model and geometry protocol (2026-10-24)

- Status: `EXP13_PHASE1A_COMPLETED_PENDING_CHATGPT_GATE`.
- Deterministic TRAIN-metadata protocol audit completed without downloading/loading/running a Diffusion model. Polygon edit-core, transition halo, immutable background, FOV uncertainty, clamped ROI, non-square canvas mapping, inverse mapping, order-independent grouping, deterministic overlap ownership, and explicit `ROI_COMPOSITION_HOLD` are implemented and tested.
- Deterministic tests: 8/8 direct assertion tests PASS; result JSON/CSV schema checks PASS. Real frozen TRAIN pilot metadata verifies Burn source `50`; boundary and thin-Crack candidates are recorded with status/risk fields. No image decode, VAL/TEST access, raw-data write, model call, generation, training, SSL, pair freeze, or Exp13.2-B authorization.
- Candidate model and configuration are audit-only. Online license/revision/mask/cache facts remain unverified and runtime fields remain `NEEDS_RUNTIME_SMOKE`; no generated images are stored. Budget counters separate planned/actual/retry calls and block local calls above the 300 cap; GPT fallback remains unauthorized.
- Evidence: `results/exp13_phase1a/20261024T_PHASE1A_PROTOCOL/`; docs: `docs/exp13_phase1a_model_protocol.md`, `docs/exp13_phase1a_geometry_design.md`, `docs/exp13_phase1a_smoke_pilot_protocol.md`, `docs/exp13_phase1a_handoff.md`.

### Exp13 Phase 1-A-R1 implementation repair (2026-10-24)

- Status: `EXP13_PHASE1A_R1_COMPLETED_PENDING_CHATGPT_GATE`.
- Evidence: `results/exp13_phase1a_r1/20261024T113936Z/`; implementation review and handoff are `docs/exp13_phase1a_r1_implementation_review.md` and `docs/exp13_phase1a_r1_handoff.md`.
- Real test execution used the documented standard-library fallback because pytest was unavailable; 9 direct tests passed, exit code 0, with command/stdout/stderr captured in `test_execution.log`.
- TRAIN-only raster compatibility compared all 1,266 polygons against Phase 0-R1 Pillow `ImageDraw.polygon(round(x/y), fill=1)` and retained Pillow as normative. No VAL/TEST access or raw-data writes occurred.
- Native RGB multi-ROI full-image composition is implemented and validated on real TRAIN sources including source `50` with all 7 polygons; shape/dtype, target coverage, and immutable outside-support pixels passed.
- BudgetProtocol now persists an atomic ledger, resumes STARTED calls as occupied, rejects duplicate IDs without double charging, enforces 4/120/300 calls, 100 production sources, 10 retry sources, and unauthorized GPT fallback.
- Smoke protocol is revised to 3 planned ROI calls across one single-ROI and one double-ROI TRAIN source; it remains protocol-only and was not run. No Diffusion, generation, training, SSL, or downstream authorization is implied.

### Exp13 Phase 1-A-R2 minimal engineering closure (2026-10-24)

- Status: `EXP13_PHASE1A_R2_COMPLETED_PENDING_CHATGPT_GATE`.
- R2 repaired the three bounded R1 findings: non-identity RGB re-paste evidence, strict persistent budget accounting, and independent raster-statistics category sets. Evidence: `results/exp13_phase1a_r2/20261024T123606Z/`.
- Standard-library tests passed 9/9 with captured command, interpreter, stdout/stderr and exit code. Real TRAIN RGB composition passed for sources `2`, `17`, `50` and `13`; source `50` covered all 7 polygons in 5 independent ROI groups, with 72 overlap pixels audited.
- Native output shape/dtype, polygon-level changes, immutable support outside the edit mask, padding preservation, mask/RGB synchronization and RGB order independence passed. Overlap conflicts are explicitly held when `hold_on_overlap` is requested.
- BudgetProtocol schema v2 uses a single locked ledger with RESERVED/STARTED/terminal accounting, atomic writes and restart/corruption HOLD. Caps, retry/source limits, group atomicity, GPT denial and two-process contention passed.
- Corrected independent raster sets report boundary 118 polygons/90 sources, thin-Crack-or-small 97/81 and small-area 2/1; each satisfies affected sources <= affected polygons. Historical R0/R1 evidence and the frozen manifest hash remain unchanged.
- No Diffusion/model call, generation, SSL, YOLO training, evaluation, VAL/TEST access or raw-data write occurred. Smoke remains unrun and no trusted pair, downstream training or Exp13.2-B authorization is implied.
- Review and handoff: `docs/exp13_phase1a_r2_final_engineering_review.md` and `docs/exp13_phase1a_r2_handoff.md`.

### Exp13 Phase 1-B SDXL engineering Smoke preflight (2026-10-24)

- Status: `EXP13_PHASE1B_SMOKE_HOLD` / `PREFLIGHT_HOLD`.
- The authorized Smoke did not start. The required Hugging Face metadata request for `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` timed out from `borescope7`; exact revision, license and file manifest could not be verified.
- Environment facts were recorded: RTX 3080 Ti 12 GiB with about 11.9 GiB free at probe time, Python 3.12.3, Torch 2.8.0+cu128, CUDA 12.8, Diffusers 0.35.1, Transformers 4.55.4, Accelerate 1.10.1, and about 30 GiB free on `/root/autodl-tmp`.
- No model download/load/inference, API call, fallback checkpoint, budget reservation, VAL/TEST access, or raw-data write occurred. No contact sheet was produced because no model output exists.
- R2 identifies source `2` as a one-ROI candidate and source `17` as a two-ROI candidate with zero edit-support overlap, but final source/mask hash freeze and Smoke execution remain undone.
- Evidence, review and handoff: `results/exp13_phase1b_smoke/20261024T130500Z/`, `docs/exp13_phase1b_smoke_review.md`, and `docs/exp13_phase1b_smoke_handoff.md`.

### Exp13 Phase 1-B-R1 SDXL preflight recovery (2026-10-24)

- Status: `EXP13_PHASE1B_SMOKE_HOLD` / `NETWORK_HOLD`.
- AutoDL DNS resolved Hugging Face IPv4, but IPv4 TCP 443 failed and IPv6 had no usable route. The Windows controlled host resolved DNS but its bounded HTTPS request timed out. No proxy, TLS bypass, token, certificate or third-party mirror was used.
- Exact model revision, `openrail++` license text, component/file manifest, LFS hashes and FP16 completeness remain unverified. No model download, transfer, load, inference, fallback model or budget reservation occurred.
- Runtime facts were recorded: RTX 3080 Ti 12 GiB; Python 3.12.3; Torch 2.8.0+cu128; CUDA 12.8; Diffusers 0.35.1; Transformers 4.55.4; Accelerate 1.10.1; about 30 GiB free on `/root/autodl-tmp`.
- R2 candidate geometry remains source `2` single ROI and source `17` double ROI, but final source/mask hash freeze and Smoke execution are deferred. No Pilot, pair freeze or SSL authorization is implied.
- Evidence, review and handoff: `results/exp13_phase1b_r1/20261024T132018Z/`, `docs/exp13_phase1b_r1_review.md`, and `docs/exp13_phase1b_r1_handoff.md`.

### Exp13 Phase 1-B-R2 trusted model channel validation (2026-10-24)

- Status: `NETWORK_HOLD` / `MODEL_ARTIFACT_HOLD`.
- This run was read-only network validation only. Windows and AutoDL system DNS returned different unverified answers; Google DoH was unreachable and Cloudflare DoH was reset. Hugging Face root HTTPS, fixed-revision API/resolve, and official `cdn-lfs.huggingface.co` were unreachable within bounded timeouts.
- The supplied reference revision `115134f363124c53c7d878647567d04daf26e41e` and `openrail++` label remain unverified references. No metadata, file list, weights, transfer, cache, model load, inference, or budget use occurred.
- No DNS/hosts/proxy/TLS changes, third-party mirror, token, VAL/TEST access, raw-data write, Pilot, SSL, or training occurred. Evidence and handoff: `results/exp13_phase1b_r2/20261024T134647Z/`, `docs/exp13_phase1b_r2_review.md`, and `docs/exp13_phase1b_r2_handoff.md`.

### Exp13 Phase 1-B-R3 offline SDXL artifact verification (2026-10-25)

- Status: `EXP13_PHASE1B_R3_ARTIFACT_VERIFIED_PENDING_CHATGPT_GATE`.
- User-provided Windows Hugging Face snapshot evidence identifies revision `115134f363124c53c7d878647567d04daf26e41e` and license label `openrail++`; AutoDL did not independently query Hugging Face. Server-side SHA256 verification matched all 19 files against the uploaded Windows manifest (0 size/hash mismatch; 6,941,223,330 verified bytes).
- Archive path safety, isolated extraction, required FP16 component structure, JSON parsing, safetensors header/layout inspection, and offline dependency imports passed. The old partial model directory was left untouched. Model weights and TAR remain outside Git.
- This was static-only: no `from_pretrained`, GPU model load, inference, Smoke budget reservation, VAL/TEST access, raw-data write, SSL, or YOLO training. Diffusers runtime load and 12 GiB Smoke feasibility remain unestablished.
- Evidence: `results/exp13_phase1b_r3/20261025T042354Z/`; review and handoff: `docs/exp13_phase1b_r3_artifact_review.md` and `docs/exp13_phase1b_r3_handoff.md`. Stop for ChatGPT Gate; no Smoke/Pilot/SSL authorization is implied.

### Exp13 Phase 1-B-S1 authorized GPU Smoke preflight (2026-10-25)

- Status: `EXP13_PHASE1B_S1_HOLD`; stopped before model load or inference.
- Source 2 passed its one-ROI TRAIN geometry check. Source 17 has two independent ROIs, but `roi_1` reaches x=718.008 in a 720-pixel image, clamps at the right boundary and overlaps the bright frame/FOV/overlay region; safe support is unresolved. The exact two-source plan was not changed and source 2 was not run by itself.
- R3 local model artifact recheck passed against the supplied manifest (19/19, 6,941,223,330 bytes). This did not invoke `from_pretrained` or GPU loading. Persistent ledger has zero reserved/started/actual calls; VAL/TEST and raw-data writes are zero.
- Input-only source/FOV preview is retained on AutoDL at `results/exp13_phase1b_s1/20261025T044024Z/source_fov_review_contact.png` and not committed. Review/handoff: `docs/exp13_phase1b_s1_smoke_review.md`, `docs/exp13_phase1b_s1_handoff.md`.
- Stop for ChatGPT Gate and explicit resolution of source eligibility; no substitution, Pilot, trusted-pair freeze, SSL, or downstream training is implied.

### Exp13 Phase 1-B-S2 safe source reselection and bounded GPU Smoke (2026-10-25)

- Status: `EXP13_PHASE1B_S2_COMPLETED_PENDING_CHATGPT_GATE`; three real SDXL ROI calls completed, but semantic restoration quality failed and no result is approved for training.
- Source 17 remained excluded for right-edge/FOV/annotation-overlay risk. Source 2 was used for the single-ROI call. Source 65 was the first visually approved dual-ROI candidate in numeric filename order; its supports were disjoint and native round-trip validation passed.
- Pinned local SDXL Inpainting loaded offline. GPU peak was 5,764,813,824 bytes allocated / 6,247,415,808 reserved. Calls took 13.489, 11.473 and 10.400 seconds. The durable S2 ledger records 3/4 Smoke calls, zero retries and one remaining call; future phases must inherit it.
- Native output shapes/dtypes and paste-back checks passed; changed pixels outside edit support were 0 for both images. Visual inspection found a black rectangular artifact in source 2 and dark/implausible output in source 65's small ROI; both outputs are rejected as pseudo-normal samples.
- Audit traceability caveat: the safe-candidate JSON is still `VISUAL_REVIEW_REQUIRED` with null selection, while the frozen pre-inference config and source manifest identify sources 2 and 65 and include source/hash provenance. The malformed process exit-code capture is preserved and not interpreted as success; stdout and terminal ledger report completion.
- TRAIN-only; VAL/TEST access 0/0, raw-data write 0, SSL/YOLO training 0. Generated images/contact sheets and all model files remain outside Git. See [S2 review](exp13_phase1b_s2_smoke_review.md), [handoff](exp13_phase1b_s2_handoff.md), and `results/exp13_phase1b_s2/`.
- Stop for ChatGPT Gate. No Pilot, batch repair, trusted-pair freeze, SSL or downstream training is authorized.

### Exp13 Phase 1-B-S3 zero-inference artifact root-cause audit (2026-10-25)

  - Status: `EXP13_PHASE1B_S3_DIAGNOSIS_COMPLETED_PENDING_CHATGPT_GATE`; CPU-only audit of persisted S2 outputs, with zero new model loads and zero new inference calls.
  - Source 2's black bar first appears in raw SDXL Canvas output: 1,172 new-dark pixels versus input, all inside edit support and 1,154 inside the original polygon. Source 65 roi_1 has 810 new-dark raw-output pixels. Inverse mapping and hard paste do not create these artifacts.
  - Manual review confirms source 2's full context includes GE/UI, physical metal boundaries, dark background and defect; source 65 roi_0 has repeated reflect-padded metal edges; source 65 roi_1 is 66x63 native pixels enlarged 15.515x. These are eligibility risks beyond white-mask overlap.
  - S2 ledger remains authoritative at 3 occupied calls of 4, with one slot held. No output is approved as a pseudo-normal pair. Selection reconciliation and malformed exit-code evidence are preserved without rewriting history.
  - See [S3 review](exp13_phase1b_s3_root_cause_review.md), [handoff](exp13_phase1b_s3_handoff.md), and `results/exp13_phase1b_s3/20261025T141700Z/`. No Pilot, batch repair, trusted-pair freeze, SSL or training is authorized.
