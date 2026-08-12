# Project state

| Field | Current fact |
|---|---|
| Current phase | Exp02.2a early-validation-loss NaN root-cause probe active by explicit user authorization |
| Last completed experiment | Exp02.2 baseline size and error audit (val-only) |
| Current data version | `/root/autodl-tmp/borescope-new-seg-data/v1` from raw manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB; smoke PASS |
| Current best model | Exp02.1 provisional baseline `best.pt`; SHA256 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`; Gate not passed |
| Current primary metric | independent VAL mask mAP50-95 = 0.29898123412927496 |
| Active blockers | Epoch 1--5 val box/seg/cls/dfl losses are NaN; explicit no-NaN Baseline Gate condition violated. Real acquisition IDs also remain unavailable. |
| Current Gate | Dataset/Environment/Smoke Gates PASS; Exp02 Baseline Gate STOP pending user review |
| Next allowed experiment | Exp02.2a diagnostic only; then stop for human review. Exp02.3/Exp03 remain prohibited |
| Formal training allowed | Only the six-epoch diagnostic reproduction configured with epochs=100 and callback stop after epoch 6; no full rerun |
| Exp01 execution commit | `dc8cd55383112b48113545ea86532a157531475e` |
| Exp01 final documentation commit | `d55cdb555c37099753ca8e1a55a456e3f0455818` |
| Exp02 frozen batch | 32; full-batch probe reserved-memory headroom 72.27%; full smoke headroom 70.34% |
| Exp02 execution commit | `d9df528383b63eeac48c0b76f0b6bd887ef8693b` (pre-documentation audit/tooling HEAD) |
| Last updated time | 2026-08-12T22:25:00+08:00 |

## Next allowed work

Wait for user review of `docs/handoffs/EXP02_BASELINE_REVIEW.md`. Do not start Exp02.3 or any other experiment automatically.
