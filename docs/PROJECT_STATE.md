# Project state

| Field | Current fact |
|---|---|
| Current phase | FastTrack-B complete; stopped for review |
| Last completed experiment | Exp07 Stage2 (NEGATIVE); Exp06 SupCon positive as ROI classifier |
| Current data version | `/root/autodl-tmp/borescope-new-seg-data/v1` from raw manifest `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |
| Current split hash | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Current environment | Project `.venv`: Python 3.12.3, torch 2.8.0+cu128, Ultralytics 8.4.117, OpenCV-headless 5.0.0; one visible RTX 2080 Ti, ~22GB; smoke PASS |
| Current best model | Baseline: Exp02.1 `best.pt` SHA256 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`; positive candidate: Exp05 hard-mining Treatment; no final best declared |
| Current primary metric | independent VAL mask mAP50-95 = 0.29898123412927496 |
| Active blockers | Case C confirmed: epoch 1--5 training-validation AMP C2PSA qk matmul overflows before loss; explicit no-NaN Baseline Gate condition remains violated. Real acquisition IDs also remain unavailable. |
| Current Gate | Exp02 Baseline Gate PASS_WITH_NUMERICAL_WAIVER; FastTrack-A complete |
| Next allowed experiment | None automatically; FastTrack-C requires explicit authorization |
| Formal training allowed | None; Exp02.2a short reproduction is complete and no full rerun is authorized |
| Exp01 execution commit | `dc8cd55383112b48113545ea86532a157531475e` |
| Exp01 final documentation commit | `d55cdb555c37099753ca8e1a55a456e3f0455818` |
| Exp02 frozen batch | 32; full-batch probe reserved-memory headroom 72.27%; full smoke headroom 70.34% |
| Exp02 execution commit | `d9df528383b63eeac48c0b76f0b6bd887ef8693b` (pre-documentation audit/tooling HEAD) |
| Exp02.2a root cause | `Case C / MODEL_FORWARD_NUMERICAL_INSTABILITY`: FP16 C2PSA attention qk matmul overflow -> softmax NaN; FP32 finite |
| Exp02.2a test discipline | `test_accessed=false` |
| Last updated time | 2026-08-13T15:00:00+08:00 |

## Next allowed work

Wait for user review of `docs/handoffs/FASTTRACK_B_REVIEW.md`. Do not start FastTrack-C or access test automatically.
