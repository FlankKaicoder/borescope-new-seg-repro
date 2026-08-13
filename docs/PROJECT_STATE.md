# Project state

| Field | Current fact |
|---|---|
| Current phase | `FAST_REPRO_COMPLETE` / waiting Final Verify authorization |
| Last completed experiment | Exp09.2a status calibration: transfer mechanism PASS; Exp09 SSL `INVALID_BY_BACKBONE_NO_UPDATE`; downstream `NOT_RUN_BY_GATE` |
| Current data version | `/root/autodl-tmp/borescope-new-seg-data/v1` from raw manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB |
| Current baseline | Exp02 YOLO11n-seg; independent VAL mask mAP50-95 `0.29898123412927496`; `PASS_WITH_NUMERICAL_WAIVER` |
| Current candidate | Exp05 Hard Mining Treatment; VAL mask mAP50-95 `0.311318`; candidate only, not a final best model |
| Current Gate | Fast Repro `COMPLETE` |
| Active blockers | None for Fast Repro closure. Final Verify is awaiting explicit user authorization. |
| Documented limitations | Exp02 early AMP validation numerical overflow is covered by `PASS_WITH_NUMERICAL_WAIVER`; Exp08 is engineering-gated and not evaluated; Exp09 SSL left 0/120 trainable backbone parameters changed and downstream is not evaluated. |
| Next allowed experiment | None automatically; Exp10 requires explicit user authorization |
| Formal training allowed | None until the user authorizes Exp10 |
| Exp01 execution commit | `dc8cd55383112b48113545ea86532a157531475e` |
| Exp01 final documentation commit | `d55cdb555c37099753ca8e1a55a456e3f0455818` |
| Exp02 frozen batch | 32; full-batch probe reserved-memory headroom 72.27%; full smoke headroom 70.34% |
| Exp02 execution commit | `d9df528383b63eeac48c0b76f0b6bd887ef8693b` (pre-documentation audit/tooling HEAD) |
| Test discipline | `test_accessed=false` throughout Fast Repro |
| Last updated time | 2026-08-13T19:45:00+08:00 |

## Next allowed work

Wait for explicit user authorization before Exp10. If authorized, compare only Exp02 Baseline and Exp05 Hard Mining across three seeds, then freeze one candidate before any one-time Exp11 TEST. Do not start Exp10/Exp11 or access TEST automatically.
