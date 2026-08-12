# Project state

| Field | Current fact |
|---|---|
| Current phase | Exp00 audit complete / Data Gate STOP |
| Last completed experiment | Exp00.5 near-duplicate conflict review package |
| Current data version | Raw dataset audit manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | NOT CREATED |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB; smoke PASS |
| Current best model | NONE |
| Current primary metric | NOT ESTABLISHED; planned primary metric is mask mAP50-95 after baseline |
| Active blockers | 24 unpaired-image semantics pending; 22 near-duplicate conflict-group label decisions pending |
| Current Gate | Data Gate STOP; Environment Gate PASS |
| Next allowed experiment | Human decisions and reviewed annotation remediation only; Exp01 remains disallowed |
| Formal training allowed | NO |
| Git HEAD | Exp00.4/00.5 implementation baseline `584474c`; use `git rev-parse HEAD` for the final documentation commit |
| Last updated time | 2026-08-12T19:09:55+08:00 |

## Next allowed work

Only human review decisions, reviewed annotation remediation planning, and Gate updates are allowed. Do not create the final split or YOLO dataset and do not train a model.
