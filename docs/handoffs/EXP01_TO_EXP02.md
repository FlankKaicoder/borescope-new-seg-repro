# EXP01 → EXP02 handoff

Generated: 2026-08-12 Asia/Shanghai  
Authoritative status: **Dataset Freeze Gate PASS / Exp02 allowed YES / Exp02 not started**

## Policy closure

- Annotation authority: PASS. Professional JSON annotations are authoritative GT.
- 24 unpaired images: `excluded_unpaired`; not background, no empty JSON, no split membership.
- 22 cross-label near-duplicate groups: all 74 images and original professional labels retained; no remediation; every connected group remains in one split.

## Frozen dataset

| Item | Value |
|---|---|
| Dataset v1 | `/root/autodl-tmp/borescope-new-seg-data/v1` |
| supervised images / instances | 969 / 1847 |
| train | 668 images / 1266 instances |
| val | 154 images / 296 instances |
| test | 147 images / 285 instances |
| classes | Burn, Crack, Dent, Material missing, Tears, Tip curl, corrosion |
| class mapping SHA256 | `d4df5e02e5eb1306d0b277c336ce413b54be1d9ce090386e2988b750da285d40` |
| split manifest SHA256 | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| raw manifest SHA256 | `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |

## Per-class instances

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Burn | 287 | 81 | 58 | 426 |
| Crack | 95 | 24 | 21 | 140 |
| Dent | 128 | 37 | 37 | 202 |
| Material missing | 184 | 32 | 33 | 249 |
| Tears | 45 | 10 | 11 | 66 |
| Tip curl | 27 | 3 | 5 | 35 |
| corrosion | 500 | 109 | 120 | 729 |

## Freeze Gate evidence

- conversion errors 0; invalid YOLO polygons 0; unexpected unmatched supervised samples 0;
- near-duplicate and SHA256 cross-split leakage 0;
- class mapping and split manifest frozen with hashes;
- 50 final YOLO-roundtrip overlays cover all seven classes and pass manual alignment review;
- Ultralytics segmentation dataset full scan and sample load PASS for train/val/test;
- real acquisition IDs are unavailable; visual near-duplicate connected components remain the documented proxy limitation.

## Git commits

- Exp01 execution commit: `dc8cd55383112b48113545ea86532a157531475e` (final execution/smoke implementation before documentation).
- Exp01 final documentation commit: the commit that adds this handoff and Exp01 final Markdown; report exact hash after commit.
- Final Git HEAD: obtain with `git rev-parse HEAD` after the finalization commit and report separately; do not call either value a generic “final commit”.

## Next boundary

Exp02 is technically allowed by the Dataset Freeze Gate, but the user required this cycle to stop. Do not start YOLO11n-seg smoke or formal baseline training until explicit authorization.
