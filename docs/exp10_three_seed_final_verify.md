# Exp10 three-seed Final Verify

Status: **EXP10_STOPPED_BY_SEED43_TREATMENT_TRAIN_NONFINITE**. `test_accessed=false`.

Seed42 reuse audit passed: baseline/control/treatment checkpoints exist and match the frozen definitions; the two arms each used 20,040 samples and 331 optimizer steps. Seed43 baseline completed 100/100 from official COCO initialization (best SHA256 `09f115c80de4d624f4fb36ee8ede1a65a372cce395d963a8e02e4b6bd65e732c`). Its TRAIN-only hard pool contained 201/668 images. Seed43 Uniform Control completed 30/30 with 20,040 samples and 331 steps (best SHA256 `8a1b785ecc3125a0d34b4c120d04aca10a46d7296c6a0b763b7b3629442924cd`).

Seed43 Hard Treatment did not complete. The first saved epoch contains non-finite train box/seg/class/dfl losses; recovery attempts remained numerically invalid and returned to NaN. This is train-loss failure, not the existing early validation-loss waiver. The process was stopped, the original client-disconnect partial directory was preserved, and the failed retry was not accepted as a model result.

Per the hard gate, seed44, unified VAL/fixed-point/size evaluation, paired deltas, aggregate statistics, robustness recommendation, and Candidate Freeze recommendation were not run. Required CSV schemas are preserved with `N/A` and `NOT_RUN_BY_GATE`; no pseudo-statistics were produced. Exp10.2 is `SKIPPED_BY_CANDIDATE_GATE`; Exp11 remains forbidden and unexecuted.
