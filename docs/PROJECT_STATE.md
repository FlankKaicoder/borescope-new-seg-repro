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
