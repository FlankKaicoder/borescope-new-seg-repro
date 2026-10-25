# Exp13 Phase 1-B-S4 handoff

**Status:** `RESTORATION_ELIGIBILITY_PREPARED_AWAITING_HUMAN_REVIEW`

S4 performed a deterministic CPU-only scan of the frozen TRAIN split. The manifest hash matched; 668 TRAIN images and 1,266 polygons were read. No VAL/TEST image, model, GPU, fourth Smoke call, raw-data write, training or SSL was used.

Results: 816 polygons from 530 independent sources pass the geometry proxy (no clamp, edge distance >=24 px, no support overlap). This is not semantic restoration approval. At representative exploratory thresholds, `scale<=4x` and valid Canvas content `>=75%` leave 26 polygons from 26 sources. Twenty deterministic contact sheets were prepared across all seven labels and 20 distinct sources; every record is `PENDING_HUMAN_REVIEW`, with machine selection null.

S2 connected-component precision is recorded separately. Source 2 has 9 components (largest 917 pixels), with 1,172/1,172 threshold pixels in support and 1,154/1,172 in the polygon. Source 65 roi_1 has one 810-pixel component, all in support and 240/810 in the polygon. Historical S3 evidence was not rewritten.

The proposed gate requires full-context UI/FOV review, normal-metal continuity, physical-boundary review, native information/scale, padding/mirror-risk accounting, geometry/mask PASS, and a non-null machine-readable selection before any future inference. Current recommendation is to wait for human review. The fourth Smoke call remains unused and must not be spent automatically.

Evidence:

- `results/exp13_phase1b_s4/20261025T075800Z/train_eligibility_distribution.json`
- `results/exp13_phase1b_s4/20261025T075800Z/candidate_screening.csv`
- `results/exp13_phase1b_s4/20261025T075800Z/eligibility_threshold_sensitivity.json`
- `results/exp13_phase1b_s4/20261025T075800Z/candidate_review_manifest.json`
- `results/exp13_phase1b_s4/20261025T075800Z/restoration_eligibility_gate.json`
- `results/exp13_phase1b_s4/20261025T075800Z/artifact_connected_component_addendum.json`
- `results/exp13_phase1b_s4/20261025T075800Z/alternative_method_feasibility.json`
- `results/exp13_phase1b_s4/20261025T075800Z/budget_preservation.json`
