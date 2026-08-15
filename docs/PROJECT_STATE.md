# Project state

| Field | Current fact |
|---|---|
| Current phase | `CANDIDATE_FROZEN_READY_FOR_EXP11` |
| Last completed formal experiment | Exp10 controlled three-seed verification: `HARD_MINING_NOT_CONFIRMED` |
| Last completed project work | Review10.5 New Dataset Full Experiment Review; training=false; test_accessed=false |
| Current data version | `/root/autodl-tmp/borescope-new-seg-data/v1` from raw manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB |
| Current baseline | Exp02 YOLO11n-seg; independent VAL mask mAP50-95 `0.29898123412927496`; `PASS_WITH_NUMERICAL_WAIVER` |
| Option decision | `OPTION_A_DIRECT_FINALIZATION` |
| Current candidate | Final frozen YOLO11n-seg Baseline; seed44; VAL Mask mAP50-95 `0.3251567516159603` |
| Final checkpoint | `/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt`; SHA256 `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9` |
| Current Gate | Candidate Freeze PASS before TEST; only Exp11 frozen TEST is allowed |
| Active blockers | All training and model/seed/threshold selection are forbidden. TEST may be accessed only by the committed Exp11 frozen evaluator. |
| Documented limitations | Exp02 early AMP validation numerical overflow is covered by `PASS_WITH_NUMERICAL_WAIVER`; Exp08 is engineering-gated and not evaluated; Exp09 SSL left 0/120 trainable backbone parameters changed and downstream is not evaluated. |
| Next allowed work | Exp11 frozen TEST only |
| Formal training allowed | NO |
| Candidate Freeze | `PASS` |
| Exp11 | `READY / TEST_NOT_ACCESSED` |
| Exp01 execution commit | `dc8cd55383112b48113545ea86532a157531475e` |
| Exp01 final documentation commit | `d55cdb555c37099753ca8e1a55a456e3f0455818` |
| Exp02 frozen batch | 32; full-batch probe reserved-memory headroom 72.27%; full smoke headroom 70.34% |
| Exp02 execution commit | `d9df528383b63eeac48c0b76f0b6bd887ef8693b` (pre-documentation audit/tooling HEAD) |
| Test discipline | `test_accessed=false` throughout Fast Repro |
| Last updated time | 2026-08-15T17:40:54+08:00 |

## Next allowed work

Candidate Freeze is complete. Only `scripts/exp11_final_test.sh` may access TEST, using the frozen seed44 Baseline checkpoint. No training, sweep, comparison, or TEST-driven selection is allowed.
