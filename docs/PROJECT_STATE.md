# Project state

| Field | Current fact |
|---|---|
| Current phase | `EXP10_TRUE_FP32_DIAGNOSTIC_COMPLETE_WAITING_REVIEW` |
| Last completed experiment | Exp10.1b TRUE-FP32 / Trainer-path no-update verification: `CASE B REPLAY_OR_PREPROCESS_PIPELINE_BUG`, subtype `NONREPRODUCIBLE_EXP10_1A_DIAGNOSTIC_REPLAY` |
| Current data version | `/root/autodl-tmp/borescope-new-seg-data/v1` from raw manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB |
| Current baseline | Exp02 YOLO11n-seg; independent VAL mask mAP50-95 `0.29898123412927496`; `PASS_WITH_NUMERICAL_WAIVER` |
| Current candidate | Exp05 Hard Mining Treatment; VAL mask mAP50-95 `0.311318`; candidate only, not a final best model |
| Current Gate | Exp10.1b supersedes the old Exp10.1a CASE C conclusion: TRUE-FP32, TRUE-AMP, legacy replay rerun, and formal Trainer path are finite; Exp10 remains stopped/incomplete |
| Active blockers | Prior Exp10.1a non-finite replay evidence is not reproducible; three-seed comparison is incomplete and a future unified numerical protocol requires explicit review/authorization. |
| Documented limitations | Exp02 early AMP validation numerical overflow is covered by `PASS_WITH_NUMERICAL_WAIVER`; Exp08 is engineering-gated and not evaluated; Exp09 SSL left 0/120 trainable backbone parameters changed and downstream is not evaluated. |
| Next allowed experiment | None automatically; review Exp10.1b CASE B replay-side evidence |
| Formal training allowed | None; Exp10 stopped and Exp11 forbidden |
| Exp01 execution commit | `dc8cd55383112b48113545ea86532a157531475e` |
| Exp01 final documentation commit | `d55cdb555c37099753ca8e1a55a456e3f0455818` |
| Exp02 frozen batch | 32; full-batch probe reserved-memory headroom 72.27%; full smoke headroom 70.34% |
| Exp02 execution commit | `d9df528383b63eeac48c0b76f0b6bd887ef8693b` (pre-documentation audit/tooling HEAD) |
| Test discipline | `test_accessed=false` throughout Fast Repro |
| Last updated time | 2026-08-15T13:47:16+08:00 |

## Next allowed work

Review `docs/handoffs/EXP10_1B_TRUE_FP32_REVIEW.md`. Do not resume Exp10, run seed44, start Exp11, freeze a candidate, or access VAL/TEST automatically.
