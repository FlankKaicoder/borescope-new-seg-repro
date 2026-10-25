# Exp13 Phase 1-B-S3 handoff

**Status:** `EXP13_PHASE1B_S3_DIAGNOSIS_COMPLETED_PENDING_CHATGPT_GATE`

S3 audited the existing S2 artifacts with CPU-only Pillow/standard-library processing. New model loads and inference calls are both zero. The S2 ledger remains authoritative: 3 actual/occupied calls, 1 Smoke call remaining, no retries, no GPT fallback.

The source-2 black bar first appears in raw SDXL Canvas output at `[251,453,514,685]`: 1,172 new-dark pixels versus input, all inside edit support and 1,154 inside the original polygon. Source 65 roi_1 has 810 new-dark raw-output Canvas pixels. Inverse mapping and hard paste do not create these artifacts.

The manual visual addendum is incorporated. Source 2's input context contains a GE UI icon, physical metal boundaries, dark background and defect content. Source 65 roi_0 has a 453x101 ROI with 77.73% reflect padding and repeated metal-edge structures. Source 65 roi_1 is only 66x63 native pixels, enlarged 15.515x, with limited independent normal-metal context. These are eligibility risks beyond white-mask overlap.

Do not attribute the problem to FP16 VAE behavior without direct evidence. The outputs are finite/non-constant. The recommended disposition is to hold the final Smoke slot. If a future Gate requires one controlled check, change only padding policy on source 2. Do not combine padding, ROI, prompt, strength, mask and precision changes.

The historical selection-null audit, malformed S2 exit-code file, and 3/4 ledger are preserved. All postprocess comparisons and the contact sheet are diagnostic-only. No Pilot, batch repair, pair freezing, SSL or training is authorized.

Key evidence:

- `results/exp13_phase1b_s3/20261025T141700Z/source2_black_artifact_audit.json`
- `results/exp13_phase1b_s3/20261025T141700Z/source65_dark_roi_audit.json`
- `results/exp13_phase1b_s3/20261025T141700Z/canvas_mask_scale_audit.json`
- `results/exp13_phase1b_s3/20261025T141700Z/human_visual_review_addendum.json`
- `results/exp13_phase1b_s3/20261025T141700Z/root_cause_hypotheses.json`
- `results/exp13_phase1b_s3/20261025T141700Z/restoration_eligibility_gate_proposal.json`
- `results/exp13_phase1b_s3/20261025T141700Z/selection_provenance_reconciliation.json`
- `results/exp13_phase1b_s3/20261025T141700Z/budget_preservation.json`
