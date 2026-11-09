# Exp13 G3-D2 final closeout: bounded partial human evidence

Final state: `G3D2_CLOSED_BOUNDED_PARTIAL_HUMAN_EVIDENCE`.
`FORMAL_TRE_GATE=NOT_ESTABLISHED`; `TRUSTED_PAIR_PASS=0`.

The user ended mechanical-landmark annotation. No further completion of the
120 planned slots is required. This report supersedes the benchmark-ready
state; the original benchmark, initial review data and preparation report are
retained as historical artifacts. This closes the human benchmark study,
not a successful formal geometry or training-Pair Gate.

## File verification and evidence scope

The user-specified private CSV was read from
`E:\borescope_review_assets\g3d2_mechanical_benchmark_20261109_release_verified\g3d2_landmarks.csv`.
It has 8,994 bytes, SHA256
`70cbb9ea26d026de6e37c8497b0a1982ba642bdc87d2f4235709d5659c62a5b6`.
The original bytes were checked unchanged after the audit and copied without
re-encoding to a new private closeout directory. Neither CSV nor coordinates,
reviewer identifiers, labels, or private annotations are published.

The frozen benchmark's canonical identity was independently verified as
`87112e0ea76b8f997f9be43b0e03480cedf4e8af45c38268174d9a566d7f7c69`.
CSV headers, exact Pair/point-slot sets, fixed NEAR/FAR and Polygon metadata,
finite coordinate bounds, confirmation fields and uncertainty were checked.
CSV has no embedded bundle hash: this establishes compatibility with the
frozen benchmark, not cryptographic proof of which page exported it.

| Quantity | Final count | Meaning |
|---|---:|---|
| Pair variants in frozen benchmark | 18 | Selected research cohort |
| Preset landmark slots | 120 | Planned slots, not completed evidence |
| CONFIRMED slots | 22 | User-entered confirmations across 4 Pair variants |
| PENDING slots | 98 | Retained as incomplete; no imputation |
| Rule-valid confirmed points | 15 | Pass programmatic filtering only |
| Excluded confirmed points | 7 | All NEAR-region mismatches |
| Pairs with complete required point coverage | 0 | None qualifies for full independent TRE evaluation |
| Trusted Pair PASS | 0 | No new Pair admission |

**Mechanical landmark identity has not been independently verified by an
expert.** Validity here means only agreement with program-defined rules; it
does not establish that the two clicks identify the same physical structure.
The CSV contains point annotations only. Pair-quality decisions, expert
judgments and full review JSON were not reconstructed or invented.

## Frozen rules and incomplete coverage

The audit reuses the frozen D2 geometry definitions: source points more than
2 px outside every selected Polygon, NEAR within the assigned Polygon's
frozen radius, FAR beyond all selected Polygon radii, source-pixel uncertainty
<=2, and no duplicate/near-duplicate accepted source or output points within
4 px. All 7 exclusions are NEAR-region mismatches; no thresholds or point
coordinates were changed to increase retained evidence.

| Pair | Confirmed | Rule-valid / confirmed | Valid NEAR by Polygon | Valid FAR | Why incomplete |
|---|---:|---:|---|---:|---|
| 558_V1 | 6 | 5/6 | 558_000: 2 | 3 | Needs 3 NEAR; one excluded |
| 570_V1 | 9 | 4/9 | 570_000: 0; 570_001: 1 | 3 | Neither Polygon has 3 NEAR; five excluded |
| 483_V1 | 1 | 1/1 | 483_000: 1 | 0 | Other 5 preset slots PENDING |
| 861_V1 | 6 | 5/6 | 861_000: 2 | 3 | Needs 3 NEAR; one excluded |

483 has six preset slots; 1/1 refers to its sole confirmed point, not completion
of its preset slot set. The other 14 Pair variants have no confirmed points.
The retained points cover Burn, Crack and Dent only, not all six classes.

## Exploratory TRE on the same retained points

No transform was fitted, optimized or replaced. The saved C0 mapping and
four frozen D1 full-match refits were evaluated on exactly the same retained
points for each Pair. Values below are source-native **median / P90 / maximum
pixels**. They describe the observed partial point set, not global image
accuracy, formal tail guarantees, repaired-target accuracy or a method ranking.
At n=1, median/P90/max are the same single residual.

| Frozen transform | 558_V1 (n=5) | 570_V1 (n=4) | 483_V1 (n=1) | 861_V1 (n=5) |
|---|---|---|---|---|
| Saved C0 | 2.24 / 3.58 / 3.99 | 1.32 / 2.68 / 2.96 | 1.86 / 1.86 / 1.86 | 1.39 / 3.92 / 4.53 |
| D1 SIFT Similarity refit | 13.58 / 17.13 / 18.98 | 6.75 / 21.98 / 28.17 | 8.11 / 8.11 / 8.11 | 0.48 / 2.97 / 3.45 |
| D1 SIFT Affine refit | 2.35 / 3.94 / 4.22 | 1.06 / 1.79 / 1.90 | 2.05 / 2.05 / 2.05 | 1.07 / 2.36 / 3.07 |
| D1 ORB Similarity refit | 13.58 / 17.71 / 20.13 | 3.52 / 20.57 / 27.64 | 15.51 / 15.51 / 15.51 | 0.93 / 3.42 / 3.97 |
| D1 ORB Affine refit | 2.69 / 3.76 / 4.29 | 0.93 / 1.53 / 1.59 | 2.24 / 2.24 / 2.24 | 1.34 / 3.22 / 3.82 |

Near-target evidence is particularly sparse. For saved C0, the NEAR residual
median/P90/max are 2.21/2.82/2.97 px on 558 (n=2), 2.96/2.96/2.96 px on
570 (n=1 on only its second Polygon), 1.86/1.86/1.86 px on 483 (n=1), and
2.20/2.85/3.01 px on 861 (n=2). No Pair has complete per-Polygon NEAR evidence.
Private and sanitized machine summaries retain all methods' NEAR/FAR and
per-Polygon metrics without making a formal numerical PASS decision.

Saved C0 and affine refits have small observed residuals on these partial
points. Similarity refits show larger observed errors on 558/570/483 but
smaller ones on 861. These case-dependent observations do not justify
replacing the baseline or declaring a winner. Correspondence identity,
incomplete local coverage, small sample sizes and purposive selection limit
the interpretation. Formal TRE Gate remains NOT_ESTABLISHED regardless of
whether individual observed numbers fall below a prior research limit.

## Scientific state freeze

Geometry, cloud repair, local composition and training suitability remain
separate. Partial landmark residuals cannot resolve defect removal, plausible
material restoration, complete mask coverage, Hard/Feather seam/halo, UI or
Local Change shortcut risks. The earlier 914 coverage limitation and 645 V2
composite observation remain historical concerns; this CSV adds no independent
expert finding for those cases. C2 556/640 gain no new mechanical evidence.

Retain the existing **SIFT + proportional mapping / Similarity / Affine**
pipeline and hard/inward-feather baselines. Do not initiate general alignment
algorithm expansion or further 120-point collection. Reopen a targeted
alignment fix only when an explicit failure case and scientific necessity
justify it, preserving this closed evidence and independent-evaluation roles.

No six-class 60-source cloud generation or formal Local Change training is
started or authorized by this closeout. No point/label/mask/transform was
edited; source images, original professional annotations, all prior results
and permanent request locks remain untouched. API POST/GPU/training/TEST/
thesis access=0. The protected untracked G2 script retains its original hash.

## Reproduction, private records and handoff

`tools/exp13_cloud/g3/closeout_g3d2_partial.py` reads the explicit CSV and
hash-verified benchmark and emits a separate point-only audit; it does not
construct a Pair-quality review JSON. Its private output directory is
`E:\borescope_review_assets\g3d2_closeout_20261109_partial`:
`g3d2_landmarks_frozen.csv`, `partial_tre_audit_private.json`, and
`sanitized_closeout_summary.json`. Old initial/empty-review results remain
unchanged. The public closeout summary contains counts and TRE aggregates,
never raw clicks, identities, notes or Polygon coordinates.

Verification: 4 new synthetic CSV/audit tests and 8 existing D2 benchmark
tests pass. Source CSV hash and archival byte identity are verified. The
local counts exactly match the user's independent audit: 120 planned,
22 confirmed, 15 rule-valid, 7 excluded, 0 complete evaluable Pairs.

G3-D2 annotation is closed. Stop at the ChatGPT Gate and await the next
research-route decision; no additional annotation is required for this stage.
