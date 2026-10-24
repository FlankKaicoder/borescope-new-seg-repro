# Exp13 Phase 0 Read-only Data and ROI Audit

Run: `20261024T093428Z`  
Status: `EXP13_PHASE0_AUDIT_COMPLETED_PENDING_CHATGPT_GATE`

This is an independently numbered, TRAIN-only, read-only audit. The authoritative input was the frozen split manifest with SHA256 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`; only rows whose manifest `split` equals `train` were opened. The raw dataset `/root/autodl-tmp/损伤训练数据集` was read but never written.

## Results

- 668 TRAIN images and 1,266 valid polygon instances were checked; all seven classes were present.
- 844 native-coordinate ROIs were planned from expanded, image-clamped instance boxes. Mapping is invertible and no center-crop/fit operation was used.
- Class instance counts: Burn 287, Crack 95, Dent 128, Material missing 184, Tears 45, Tip curl 27, corrosion 500.
- The pilot proposal contains 21 source-unique TRAIN images, three per class, with geometry-only simple/medium/difficult labels. It is planning evidence, not a trusted-pair freeze.

## Gates

| Gate | Result | Basis |
|---|---|---|
| 0-A DATA_INTEGRITY | PASS | 668/668 source image and JSON paths resolved; manifest dimensions and hashes were checked. |
| 0-B GEOMETRY_AND_FOV | CONDITIONAL_PASS | Polygon geometry and native ROI mapping are usable; gray/padding and FOV border signals remain heuristic and are not auto-corrected. |
| 0-C CLASS_COVERAGE | PASS | All seven classes have TRAIN image and instance support. |
| 0-D BUDGET_FEASIBILITY | PASS | 21 sources x 2 strengths = 42 source-strength combinations; 25 ROIs across those sources; limits recorded as 120 smoke/pilot calls, 300 local calls, 100 sources. |
| 0-E ENVIRONMENT_AND_PRESERVATION | PASS | Server branch/HEAD, GPU, disk, package versions, manifest hash, and read-only policy are recorded. |

No VAL/TEST image was opened. No model download or inference, diffusion/inpainting, training, SSL, threshold tuning, hard mining, or synthetic generation occurred. No trusted pair, restoration quality pass, Exp13.2-B authorization, or SSL authorization is asserted.

Canonical artifacts are under `results/exp13_phase0_readonly/20261024T093428Z/`.
