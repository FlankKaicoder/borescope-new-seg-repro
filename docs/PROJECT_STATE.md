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

No required experiment remains. Future work is outside the closed experimental phase.
