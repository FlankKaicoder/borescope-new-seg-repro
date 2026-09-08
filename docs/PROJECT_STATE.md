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

### Phase C1 quality review (2026-09-08)

- Phase C1 Preview Generation Gate B remains `PASS`; Distribution Audit Gate C remains `CONDITIONAL_PASS`.
- Quality Review status is `INSUFFICIENT_EVIDENCE_NEEDS_MANUAL_REVIEW`.
- The existing review has 876 candidate records for 860 unique edits, with 16 edits carrying both candidate reasons. All `review_label` fields are blank.
- C and D retain the unresolved risks: 41.74% and 42.41% respectively below visible-change ratio 0.50, with mean connected-component counts 19.84 and 80.23.
- No generator repair is confirmed or authorized. Gate C is not `PASS`, and Proxy Training is not authorized.

### Phase C1 manual-review preparation (2026-09-08)

- Status: `PREPARED_AWAITING_HUMAN_ANNOTATION`; this is a read-only preparation package and makes no new Gate decision.
- `results/phaseC1_manual_review/` contains all 96 weak-change candidates, plus fixed-seed (`42`) samples of 100 C-family and 100 D-family noise candidates.
- The additive `manual_annotation_template.csv` contains 290 unique `(source_id, edit_id)` records after preserving six overlaps between the three selected groups. Every `human_label` and `comment` field remains blank.
- The three contact sheets provide source, edited, combined final-preview difference mask, stored-bbox context, and the requested metadata for each selected sample. The displayed difference mask is not edit-isolated evidence.
- Gate C remains `CONDITIONAL_PASS`; generator repair and Proxy Training remain unauthorized pending completed human annotation and a later read-only Gate decision.

### Phase C1-R1 Local Change V2 repair (2026-09-08)

- This is an independently numbered TRAIN-only generation/audit extension authorized to test C/D morphology repair; A/B and the V1 preview remain unchanged.
- Formal generation: `PASS`; 668 TRAIN images, 2004 preview pairs, seeds `42,1,0`, 4052 accepted edits, 442 rejected candidates, zero exhausted edits, VAL/TEST access 0/0.
- C changed from `C_elongated_ribbon` to `C_crack_like_structural_evolution`; accepted audit mean components 1.00 and visible-change-ratio-below-0.5 fraction 0.00.
- D changed from `D_irregular_boundary` to `D_boundary_evolution`; accepted audit mean components 1.00 and visible-change-ratio-below-0.5 fraction 0.00.
- Read-only audit: `CONDITIONAL_PASS`; remaining candidates are 2 weak and 13 noise under the frozen combined-preview audit, with one image-level ratio above 0.30. This is evidence of C/D improvement, not a global Gate C promotion.
- R1 stops after audit. Proxy Training, SSL pretraining, downstream evaluation, split changes, and further repair are not executed.

### Phase C1-R1 semantic quality review preparation (2026-09-08)

- Status: `PREPARED_AWAITING_HUMAN_ANNOTATION`; this is a read-only semantic-review preparation package built only from the completed R1 formal V2 artifacts.
- Fixed seed `42` selected 100 unique-source family C edits, 100 unique-source family D edits, and 50 unique-source samples for each A/B sanity group. All 300 `human_label` and `comment` fields are blank.
- Contact sheets show saved source/final preview panels and a combined final-preview difference mask around the stored edit bbox. The completed run did not retain isolated per-edit masks, so no automatic edit-level semantic conclusion is made.
- R1 formal artifacts do not contain a source-to-class mapping. `class` is recorded as `CLASS_PROVENANCE_UNAVAILABLE`; seven-class coverage is not asserted.
- Gate C remains `CONDITIONAL_PASS`. Human annotation and a later read-only Gate decision are required before any Proxy Training consideration.
