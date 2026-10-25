# Exp13 Phase 1-B-S4 Restoration Eligibility Feasibility Gate

Run: `20261025T075800Z`
Status: `RESTORATION_ELIGIBILITY_PREPARED_AWAITING_HUMAN_REVIEW`

## Scope and protection

This was a CPU-only, TRAIN-only feasibility audit. It read the frozen manifest (`SHA256 35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`) and the 668 TRAIN images/annotations. It did not read VAL/TEST, modify the raw dataset, load a model, call a GPU, consume the fourth Smoke slot, or create training data. The S2 ledger remains 3 actual/occupied calls with one Smoke call remaining.

The implementation is `tools/exp13_restoration/run_phase1b_s4_eligibility.py`. Machine evidence is under `results/exp13_phase1b_s4/20261025T075800Z/`; the 20 contact sheets remain on the controlled AutoDL server.

## Distribution

| Measure | Count |
|---|---:|
| TRAIN images | 668 |
| TRAIN polygons | 1,266 |
| Geometry-eligible polygons | 816 |
| Geometry-eligible independent sources | 530 |
| Automatic restoration-eligible | 0 |
| Human-review contact sheets | 20 |
| Contact-sheet independent sources | 20 |

Geometry eligibility means no ROI clamp/image-edge condition, polygon edge distance at least 24 px, and no unresolved support overlap. It does not mean semantic restoration eligibility.

Geometry-proxy counts by class are: corrosion 309/500, Burn 133/287, Material missing 128/184, Dent 109/128, Crack 78/95, Tears 34/45, and Tip curl 25/27. These are screening counts only. The larger counts for corrosion, Burn and Material missing do not establish that their restoration semantics are safer.

## Threshold sensitivity

The thresholds are exploratory engineering filters, not scientific standards. Representative results:

| Max scale | Minimum valid Canvas content | Minimum native short side | Polygons / sources |
|---:|---:|---:|---:|
| 2x | 60% | 16 px | 1 / 1 |
| 4x | 60% | 16 px | 49 / 48 |
| 4x | 75% | 16 px | 26 / 26 |
| 8x | 60% | 16 px | 182 / 162 |
| 8x | 75% | 16 px | 116 / 110 |

The 16/32/64 px minimum-short-side variants were identical for these retained sets. This is a property of this dataset and transform, not evidence that the thresholds are validated.

## Gate checks

- **REG-1 UI/FOV/gray border:** border/clamping is automatically held; non-border UI, logo, bright edge, dark background and FOV semantics remain `UNKNOWN_HUMAN_REVIEW_REQUIRED`. Full-context review is mandatory. Brightness thresholds were not used to declare invalid FOV.
- **REG-2 normal-metal context:** context pixel count, connected components, boundary and luma statistics are proxy metrics only. They are not normal-metal semantic recognition. All candidates remain human-review required.
- **REG-3 physical structure boundary:** seam, hole, weld, shadow/entity boundary and occlusion status are unknown without visual review; unresolved cases are HOLD.
- **REG-4 native information/scale:** native ROI dimensions, polygon/support/context pixels, scale and Canvas content fraction are recorded. Scale/content thresholds are exploratory filters.
- **REG-5 padding:** all four padding sides, ratio and mirror boundary are recorded. Mask absence from padding is not treated as context safety. High padding is a risk flag.
- **REG-6 geometry/mask:** the audited geometry implementation records polygon/support hashes, full polygon coverage, white=redraw convention, non-clamped ROI status, overlap and native/Canvas transform.
- **REG-7 selection:** `selection=null`; no human review was performed, so no source/ROI is frozen or eligible for inference.

## Human-review packet

Twenty deterministic sheets cover all seven labels represented among geometry candidates: corrosion (2), Burn (3), Material missing (3), Crack (3), Dent (3), Tip curl (3), and Tears (3). They come from 20 distinct TRAIN sources. Each sheet contains original RGB, polygon/ROI overlay, native ROI, support mask, CPU-only expected Canvas, padding/content boundary, and proxy statistics. It is explicitly not a model-generated result.

Server directory:

`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s4/20261025T075800Z/candidate_contact_sheets/`

No candidate has `RESTORATION_ELIGIBLE` status before this human review.

## S3 connected-component precision addendum

The existing S3 report remains unchanged and was bbox-only. This run adds connected components from the persisted S2 PNGs. Source 2 threshold set (`output luma <=25`, input luma `>=45`) has 1,172 pixels, 9 eight-connected components, and largest component area 917 at `[477,478,514,519]`; 100% maps to support and 98.464% to the original polygon. Source 65 roi_1 has 810 pixels, one component at `[278,533,309,569]`; 100% maps to support and 29.630% to the polygon. No model call was made.

## Future route comparison

Route A, SDXL local restoration, can represent texture and lighting but currently has raw-Canvas artifacts, context contamination risk and semantic hallucination. It remains HOLD. Route B, deterministic local texture repair/transfer, is worth a separate CPU-only control because it is reproducible and does not consume GPU budget; it cannot infer missing geometry and must not be treated as trusted pseudo-normal data. No classical repair was executed in S4.

The last Smoke call has no current scientific use: one call cannot resolve the eligibility risks or establish semantic quality after two raw-output failures. It remains reserved and unused.

## Gate recommendation

`RESTORATION_ELIGIBILITY_PREPARED_AWAITING_HUMAN_REVIEW`. There are far more than three independent exploratory candidates, but zero restoration-eligible candidates until full-context UI/FOV, normal-metal continuity, structure-boundary and selection review are completed. Do not start the fourth Smoke call, Pilot, batch generation, pair freeze, SSL or training.
