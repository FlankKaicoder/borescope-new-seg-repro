# Exp13 Phase 1-B-S4-R1 Strict-26 Review Preparation

Run: `20261025T171500Z_s4r1`  
Status: `EXP13_PHASE1B_S4_R1_STRICT26_REVIEW_PREPARED_PENDING_CHATGPT_GATE`

## Scope and protection

This run is a CPU-only, TRAIN-only audit and review-package preparation. It
re-read the persisted S4 screening CSV and threshold-sensitivity JSON, then
read only frozen TRAIN image/annotation files to render review material. It
did not import or load a diffusion model, use a GPU, consume the fourth Smoke
slot, access VAL/TEST, write the raw dataset, run training/SSL/YOLO, or select
a trusted pair. The historical S4/S2/S3 evidence was not modified.

## Independent screening result

The strict predicate was reapplied to every persisted S4 row:

`split=train`, `geometry_status=GEOMETRY_ELIGIBLE`, `scale<=4`,
`content_fraction>=0.75`, `native_short_side>=16`, `roi_clamped=false`,
`edge_distance_px>=24`, and no `UNRESOLVED_SUPPORT_OVERLAP`.

The result is exactly **26 polygons from 26 independent sources**:

| Class | Count |
|---|---:|
| Material missing | 14 |
| Burn | 8 |
| corrosion | 4 |

The S4 `eligibility_threshold_sensitivity.json` independently records the
same 26/26 result for the 4x/75%/16px threshold. During sheet generation,
Phase 0-R1 Pillow polygon masks and support masks were rebuilt from the raw
annotation coordinates. All 26 polygon and support hashes matched the
persisted S4 CSV (`geometry_hash_mismatches=[]`).

The strict set has **zero intersection** with the original S4 20-sheet
manifest. This is expected because the original packet was class/source
diverse rather than a strict-threshold sample; the original manifest remains
unchanged.

## Review package

AutoDL controlled package:

`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s4_r1/20261025T171500Z_s4r1/strict26_review_package/`

It contains 26 independent contact sheets, a category overview, a README,
the pending human-review CSV and a candidate manifest. Every sheet labels the
Canvas as `CPU_INPUT_PREVIEW_NOT_MODEL_GENERATED` and marks UI/FOV, normal
metal context and structure-boundary semantics as
`UNKNOWN_REQUIRES_HUMAN_REVIEW`. The ZIP contains 30 members: 26 candidate
PNGs, one overview PNG, README, template CSV and manifest. It is 16,536,565
bytes with SHA256
`a29af7cd862ad3fe433dcaf2ea068216d95ca909208848a2af13df51a1f2bd3f`.
No model file, checkpoint or unrelated raw asset is included.

The template has 26 rows. All decision and semantic-review fields are
`PENDING` or blank. `RESTORATION_ELIGIBLE=PASS` is not assigned anywhere;
the machine selection remains null.

## Historical overlap and gate

Historical DreamShaper manifests were inspected read-only for source-ID
overlap. No strict source appeared in those manifests, and no image/annotation
hash equivalence or historical variant selection was asserted. The supplied
ChatGPT 20-sheet review remains a preliminary bridge, not Codex-independent
visual validation or formal acceptance.

The inherited S2 ledger is unchanged at 3 actual/occupied calls, one Smoke
slot remaining, zero new inference/model loads, zero GPT fallback, zero
VAL/TEST access, and zero raw-data writes. The current gate is still pending
human/ChatGPT review; this package prepares evidence but does not grant a
restoration-quality or trusted-pair pass.

Evidence: `results/exp13_phase1b_s4_r1/20261025T171500Z_s4r1/` and the
controlled AutoDL package above.
