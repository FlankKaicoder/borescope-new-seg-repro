# Project state

| Field | Current fact |
|---|---|
| Current phase | Exp00 policy closure complete / Exp01 dataset construction authorized |
| Last completed experiment | Exp00.5 near-duplicate conflict review package; blockers closed by user policy |
| Current data version | Raw dataset audit manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | NOT CREATED |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB; smoke PASS |
| Current best model | NONE |
| Current primary metric | NOT ESTABLISHED; planned primary metric is mask mAP50-95 after baseline |
| Active blockers | None for Exp01. The 24 unpaired images are `excluded_unpaired`; all professional JSON labels are authoritative GT. |
| Current Gate | Annotation Authority PASS; Data Gate PASS FOR EXP01; Environment Gate PASS |
| Next allowed experiment | Exp01.0--Exp01.2 dataset construction and Dataset Freeze Gate only |
| Formal training allowed | NO |
| Git HEAD | Server execution baseline `3d7a1c56930c0a6e93d8042cc883adafc80f1673`; use `git rev-parse HEAD` for the final documentation commit |
| Last updated time | 2026-08-12T19:09:55+08:00 |

## Next allowed work

Build and validate Exp01.0--Exp01.2. Formal model training remains prohibited until the Dataset Freeze Gate passes, and this work cycle must stop before Exp02 even if it passes.
