# Project state

| Field | Current fact |
|---|---|
| Current phase | Exp01 complete / Dataset Freeze Gate PASS / stopped before Exp02 |
| Last completed experiment | Exp01.2 Dataset v1 freeze and seven-class visual/load verification |
| Current data version | `/root/autodl-tmp/borescope-new-seg-data/v1` from raw manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB; smoke PASS |
| Current best model | NONE |
| Current primary metric | NOT ESTABLISHED; planned primary metric is mask mAP50-95 after baseline |
| Active blockers | None for Exp02 entry. Real acquisition IDs remain unavailable and are documented as a leakage-control limitation. |
| Current Gate | Annotation Authority PASS; Environment Gate PASS; Data Gate PASS; Dataset Freeze Gate PASS |
| Next allowed experiment | Exp02.0 smoke only after explicit user authorization; this cycle stops here |
| Formal training allowed | NO |
| Exp01 execution commit | `dc8cd55383112b48113545ea86532a157531475e` |
| Exp01 final documentation commit | The commit containing this final Exp01 documentation; report its exact hash separately after commit. |
| Last updated time | 2026-08-12T20:38:00+08:00 |

## Next allowed work

Wait for explicit user review and authorization. Dataset Freeze Gate is PASS, so Exp02 is technically allowed, but no formal model training was started in this cycle.
