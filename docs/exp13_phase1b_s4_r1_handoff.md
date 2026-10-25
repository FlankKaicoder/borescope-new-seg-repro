# Exp13 Phase 1-B-S4-R1 Handoff

Final state: `EXP13_PHASE1B_S4_R1_STRICT26_REVIEW_PREPARED_PENDING_CHATGPT_GATE`

The strict S4 predicate was independently recomputed from the persisted
`candidate_screening.csv` and cross-checked against
`eligibility_threshold_sensitivity.json`. It yielded 26 TRAIN polygons from
26 independent sources (Material missing 14, Burn 8, corrosion 4). Rebuilt
Phase 0-R1 polygon/support hashes matched all 26 persisted records. None of
the strict candidates overlaps the original 20-sheet S4 review manifest.

AutoDL has the complete review package and ZIP at:

`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s4_r1/20261025T171500Z_s4r1/strict26_review_package/`

ZIP SHA256: `a29af7cd862ad3fe433dcaf2ea068216d95ca909208848a2af13df51a1f2bd3f`; size 16,536,565 bytes. The Git result directory contains only small JSON/CSV/README evidence; industrial PNGs and ZIP remain on the controlled server.

Human review is still required for UI/FOV contamination, normal-metal
continuity, physical structure boundaries, padding/mirror risk and semantic
restoration. All 26 template decisions are pending and restoration-eligible
count is zero. Do not run the remaining Smoke call, diffusion Pilot, batch
generation, trusted-pair freeze, SSL or YOLO training from this handoff.

Inherited S2 budget remains 3/4 occupied with one slot unused. Historical S4,
S2 and S3 artifacts are preserved. The next action is an independent visual
review/Gate decision, not automatic inference.
