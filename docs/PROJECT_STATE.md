# Project state

| Field | Current fact |
|---|---|
| Current phase | FastTrack-C closed at Exp09.2a REAL_TRANSFER_FAILURE |
| Last completed experiment | Exp09.2a REAL_TRANSFER_FAILURE; downstream NOT_RUN_BY_GATE |
| Current data version | `/root/autodl-tmp/borescope-new-seg-data/v1` from raw manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB; smoke PASS |
| Current best model | Baseline: Exp02.1 `best.pt` SHA256 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`; positive candidate: Exp05 hard-mining Treatment; no final best declared |
| Current primary metric | independent VAL mask mAP50-95 = 0.29898123412927496 |
| Active blockers | Exp09 frozen SSL export changed 0/120 trainable backbone parameters versus COCO; extension candidate is NONE. Real acquisition IDs remain unavailable. Exp02 AMP overflow is a documented limitation covered by PASS_WITH_NUMERICAL_WAIVER, not an active blocker. |
| Current Gate | Exp09.2a REAL_TRANSFER_FAILURE; FastTrack-C closed |
| Next allowed experiment | None automatically; Exp10 requires explicit user authorization |
| Formal training allowed | None; Exp09 downstream is forbidden by gate and Exp10/Exp11 are not authorized |
| Exp01 execution commit | `dc8cd55383112b48113545ea86532a157531475e` |
| Exp01 final documentation commit | `d55cdb555c37099753ca8e1a55a456e3f0455818` |
| Exp02 frozen batch | 32; full-batch probe reserved-memory headroom 72.27%; full smoke headroom 70.34% |
| Exp02 execution commit | `d9df528383b63eeac48c0b76f0b6bd887ef8693b` (pre-documentation audit/tooling HEAD) |
| Exp02.2a root cause | `Case C / MODEL_FORWARD_NUMERICAL_INSTABILITY`: FP16 C2PSA attention qk matmul overflow -> softmax NaN; FP32 finite |
| Exp02.2a test discipline | `test_accessed=false` |
| Last updated time | 2026-08-13T19:20:00+08:00 |

## Next allowed work

Wait for explicit user authorization before Exp10. Do not start Exp10/Exp11 or access TEST automatically.
