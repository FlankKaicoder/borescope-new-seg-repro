# Exp10 three-seed Final Verify

## Final controlled-restart result

Status: **EXP10_COMPLETE_WAITING_CANDIDATE_FREEZE_REVIEW / HARD_MINING_NOT_CONFIRMED**. `test_accessed=false`.

The historical seed43 failed run remains preserved. Exp10.1b invalidated only its later deterministic FP32/model-forward replay explanation (`NONREPRODUCIBLE_DIAGNOSTIC_REPLAY`). A fresh seed43 Treatment controlled restart then completed 30/30 with finite TRAIN loss and finite/reloadable checkpoints. Seed44 Baseline 100/100, TRAIN-only hard pool, Uniform Control 30/30, and Hard Treatment 30/30 all passed their numerical and fairness Gates.

| Seed | Baseline Mask mAP50-95 | Control | Treatment | Treatment-Control |
|---:|---:|---:|---:|---:|
| 42 | 0.298981 | 0.280964 | 0.311318 | +0.030354 |
| 43 | 0.299506 | 0.328399 | 0.321726 | -0.006673 |
| 44 | 0.325157 | 0.341760 | 0.301449 | -0.040311 |

Treatment-Control paired delta is positive on 1/3 seeds. Mean Mask mAP50-95 delta is `-0.005543 ± 0.035346`; mean Mask Recall delta `+0.001349 ± 0.121079`; mean fixed-F1 delta `-0.005063 ± 0.039876`; mean FP delta `-11.333 ± 23.692`; mean FN delta `+5.333 ± 18.583`.

Three-seed Mask mAP50-95 means are Baseline `0.307881 ± 0.014963`, Uniform Control `0.317041 ± 0.031950`, and Hard Treatment `0.311498 ± 0.010140`. Because the fair method effect is negative on two seeds and negative on average, the final robustness recommendation is `HARD_MINING_NOT_CONFIRMED`. Baseline is recommended for Candidate Freeze review; no Candidate Freeze was executed.

The complete tables are `results/final_verify/exp10_three_seed_summary.csv`, `results/final_verify/exp10_paired_deltas.csv`, and `results/final_verify/exp10_all_effects.csv`. All 165 checked training, unified-VAL, and aggregate images exist, are non-empty, and decode successfully.

## Historical stopped phase (preserved)

Historical status before controlled restart: **EXP10_STOPPED_BY_SEED43_TREATMENT_TRAIN_NONFINITE**. `test_accessed=false`.

Seed42 reuse audit passed: baseline/control/treatment checkpoints exist and match the frozen definitions; the two arms each used 20,040 samples and 331 optimizer steps. Seed43 baseline completed 100/100 from official COCO initialization (best SHA256 `09f115c80de4d624f4fb36ee8ede1a65a372cce395d963a8e02e4b6bd65e732c`). Its TRAIN-only hard pool contained 201/668 images. Seed43 Uniform Control completed 30/30 with 20,040 samples and 331 steps (best SHA256 `8a1b785ecc3125a0d34b4c120d04aca10a46d7296c6a0b763b7b3629442924cd`).

Seed43 Hard Treatment did not complete. The first saved epoch contains non-finite train box/seg/class/dfl losses; recovery attempts remained numerically invalid and returned to NaN. This is train-loss failure, not the existing early validation-loss waiver. The process was stopped, the original client-disconnect partial directory was preserved, and the failed retry was not accepted as a model result.

Per the hard gate, seed44, unified VAL/fixed-point/size evaluation, paired deltas, aggregate statistics, robustness recommendation, and Candidate Freeze recommendation were not run. Required CSV schemas are preserved with `N/A` and `NOT_RUN_BY_GATE`; no pseudo-statistics were produced. Exp10.2 is `SKIPPED_BY_CANDIDATE_GATE`; Exp11 remains forbidden and unexecuted.
