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
