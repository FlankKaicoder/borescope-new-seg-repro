# Project state

| Field | Current fact |
|---|---|
| Current phase | Exp02 active by explicit user authorization; scope limited to Exp02.0--Exp02.2 |
| Last completed experiment | Exp02.0 YOLO11n-seg full-batch probe and one-epoch smoke |
| Current data version | `/root/autodl-tmp/borescope-new-seg-data/v1` from raw manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB; smoke PASS |
| Current best model | NONE; Exp02.0 smoke checkpoints are pipeline evidence only |
| Current primary metric | NOT ESTABLISHED; planned primary metric is mask mAP50-95 after baseline |
| Active blockers | None for Exp02 entry. Real acquisition IDs remain unavailable and are documented as a leakage-control limitation. |
| Current Gate | Annotation Authority PASS; Environment Gate PASS; Data Gate PASS; Dataset Freeze Gate PASS; Exp02.0 Smoke Gate PASS |
| Next allowed experiment | Exp02.1 YOLO11n-seg 640 100-epoch baseline with frozen batch=32; then Exp02.2 val-only audit |
| Formal training allowed | YES for the frozen Exp02.1 640 baseline only; Exp02.3 and all later methods remain prohibited |
| Exp01 execution commit | `dc8cd55383112b48113545ea86532a157531475e` |
| Exp01 final documentation commit | `d55cdb555c37099753ca8e1a55a456e3f0455818` |
| Exp02 frozen batch | 32; full-batch probe reserved-memory headroom 72.27%; full smoke headroom 70.34% |
| Last updated time | 2026-08-12T21:49:00+08:00 |

## Next allowed work

Execute only Exp02.0--Exp02.2. Train and diagnose with train/val; test predictions and test metrics are forbidden. Stop after the Exp02 Baseline Gate and do not start Exp02.3.
