# Exp13.2-A0 historical restoration review preparation

Gate: `RESTORATION_REVIEW_ASSETS_PREPARED_AWAITING_HUMAN_REVIEW`.

R1: 40/40 outputs, 20 sources. R2: 48/48 outputs, four sources already in R1. Combined: 88 candidates / 20 source groups; Crack and corrosion each have 10 sources and 44 candidates. Variants are correlated. There are 88 distinct output file hashes; this does not establish independent information or credible pseudo-normal targets.

All 352 references to 164 unique source/mask/output files are present and decodable. All 195 files in historical R1/R2 directories retain identical SHA256. Native source/masks match in dimensions. Historical outputs are 512x512 after `ImageOps.fit` central crop and scale, RGB=LANCZOS, masks=NEAREST. Ten source defect masks are partly cropped, with native retained area 44.1285%–100%; cropped-away defects are not repaired. Crop boxes and retention fractions are recorded per candidate.

Diagnostic MAE medians (0–255), grouped by historical stage and class:

| Stage/class | Defect inside | Edit support inside | Support+8px outside | ROI bbox outside |
|---|---:|---:|---:|---:|
| R1 Crack | 26.263 | 16.727 | 5.105 | 5.056 |
| R1 corrosion | 24.937 | 18.178 | 6.546 | 6.877 |
| R2 Crack | 39.751 | 20.376 | 3.789 | 3.710 |
| R2 corrosion | 21.389 | 18.914 | 6.498 | 7.128 |

These are descriptive candidate-level diagnostics, not quality scores, tests or success rates. Definitions, halo/padding and feathering/crop/compression caveats are in [the preparation report](../../../docs/exp13_2_a0_restoration_feasibility_preparation.md).

Use [review navigation](review_navigation.md), 20 source sheets and the separate Crack/corrosion overviews. Complete the 88-row `human_quality_review_template.csv` only after actual review; all 792 human judgment cells are intentionally empty. Check defect removal, normal metal plausibility, unchanged background/geometry, hallucinated holes/components/seams, mask leakage, grey borders and smoothing. Cropped/retained defects and smooth appearance alone do not prove restoration.

Polygon-derived masks are annotation-derived supervision, not evidence of pure self-supervision. Credible pair count, source coverage and multi-variant information gain are unknown. R1/R2 generation Gates and R2-QA archival remain intact. No new model generation, inference/training, VAL/TEST image reads or dataset/paper changes occurred. Exp13.2-B and generation/training remain unauthorized. Stop for human review.
