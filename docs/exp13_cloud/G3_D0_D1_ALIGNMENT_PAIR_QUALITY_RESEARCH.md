# Exp13-G3-D0/D1 alignment and Local Change Pair quality study

Status: `G3D1_METHOD_STUDY_COMPLETE_PENDING_CHATGPT_AUDIT_GATE`.

## Question and scope

This CPU-only study asks whether the frozen G3-C0 feature-screen candidates
generalize to correspondences not used to fit their transform, whether a
second local feature family adds useful evidence, and what those checks say
about the three G3-C2 V2 outputs. It does not alter old alignments, raw images,
masks, request locks, or review decisions. It used 36 C0 pairs and the three
C2 V2 pairs (39 pair-variants; 36 unique TRAIN sources). API POST, GPU,
training, TEST access, and thesis access were zero.

## Methods and controls

The existing G3-C0 pipeline remains the production baseline: SIFT/Lowe,
Similarity and Affine RANSAC, geometry bounds, distributed support, valid
overlap checks, and target-only hard/inward-feather composites. The saved C0
candidate/unsupported decisions are retained unchanged.

The D1 validator independently extracted SIFT and ORB matches over the full
image, applied a fixed ratio test and one-match-per-24-pixel source/output
grid, then fit fixed 3-pixel RANSAC Similarity and Affine transforms. A
deterministic spatial interleave held out one fifth of source-grid cells before
fitting. Held-out source points were evaluated after mapping output points
back to source coordinates. The fixed pass requires median error no greater
than `max(2 px, 0.5% source diagonal)` and P90 no greater than twice that
limit. Sparse holdout sets are `INSUFFICIENT_SPATIAL_HOLDOUT`, never PASS.

A synthetic affine control with known scale, rotation, and translation
recovered a median corner error of 0.078 px from 211 SIFT correspondences.
This validates the coordinate convention and implementation on a controlled
case; it does not represent camera or cloud-generation behavior. No per-image
threshold tuning or nonrigid fitting was used. ECC was not run in this study:
outputs contain large appearance/content changes and sometimes different
aspect ratios, so a photometric optimizer can converge to content similarity
rather than mechanical correspondence. It remains a separately testable
candidate if a valid ROI/photo-robust initialization is established.

The saved C0 polygons/masks are not available alongside every private review
render. The validator therefore reports target-neighborhood coverage only for
the 28 C0 records with persisted valid target masks. That statistic measures
geometric footprint coverage, not local structural alignment. UI/Logo masks
and expert mechanical landmarks were not available to this computation.

## Results

The synthetic control passed. On the 36 C0 variants, the spatial holdout
classified SIFT Affine as 16 PASS, 17 FAIL, and 3 insufficient; ORB Affine as
16 PASS, 19 FAIL, and 1 insufficient. Thus the independent test exposes many
feature-screen candidates whose in-sample fit did not generalize. Among the
28 original `GEOMETRY_CANDIDATE` records, SIFT holdout produced 15 PASS, 12
FAIL, and 1 insufficient. Of the original 8 `ALIGNMENT_UNVERIFIED` records,
one unexpectedly passes SIFT holdout, five fail, and two remain insufficient.
That one diagnostic pass is not a promotion: the test uses full-frame
appearance features and still lacks independent target/UI/mechanical review.

Across all C0 variants, SIFT and ORB each have 17 held-out passes. Their
failure patterns and tail errors differ by source; ORB does not produce a
consistent gain and should remain a diagnostic cross-check, not replace SIFT.
The accepted C0 baseline methods (proportional resize, Similarity, Affine)
remain valuable where supported by independent evidence. Affine is useful for
645-like mild anisotropic scale changes, while Similarity is a simpler model
for near-uniform mappings. More flexible geometry is not automatically
better.

For the three C2 V2 variants:

| Source | Existing V2 status | SIFT Affine holdout | ORB Affine holdout | Conclusion |
|---|---|---|---|---|
| 556 | `ALIGNMENT_UNVERIFIED`, 8 original matches | 1 held-out point; insufficient | median 0.78 px, P90 82.71 px; fail | ORB's low median hides a severe tail. Keep unregistered. |
| 640 | `ALIGNMENT_UNVERIFIED`, 11 original matches | median 1.05 px, P90 316.14 px; fail | median 1.22 px, P90 24.48 px; fail | Large outlier tail and no method agreement. Keep unregistered. |
| 645 | `GEOMETRY_CANDIDATE`, 76/77 original inliers | median 1.04 px, P90 2.08 px; pass | median 0.85 px, P90 7.44 px; pass | Independent SIFT/ORB evidence supports a geometry candidate. |

Source 645 is the only C2 pair with convergent held-out evidence. This supports
its geometric mapping, not semantic repair, defect-edge fidelity, composite
quality, or training eligibility. The existing private V2 comparison album
remains the human visual review artifact.

For C0 target-neighborhood coverage, both SIFT and ORB affine transforms had
median valid footprint 1.00 across the 28 available masks (minimum 0.844 and
0.830 respectively; 27/28 at least 0.98). These high values show that the
projected raw canvas reaches the polygon neighborhood. They do not establish
that the corresponding metal edge or component is correct. The remaining
eight sources lack a persisted target mask in the review renders, so this
local metric is unavailable for them.

## Pair quality and failure interpretation

The evidence now separates four questions. Geometry asks whether independent
normal-structure correspondences support one bounded transform. Repair
quality asks whether the generated image restores the defect plausibly without
changing real boundaries. Composite quality asks whether the selected mask,
valid support, feather edge, and UI/Logo handling are acceptable. Local Change
Pair eligibility requires all three plus explicit expert review of the defect
neighborhood and confirmation that the clean target is suitable supervision.

Exact source pixels outside a hard/feather mask are a compositor invariant,
not evidence for any of those questions. Feathering can also create a
synthetic edge; keep hard and feather versions separately reviewable and do
not treat either as the raw generated image.

No source meets a complete trusted-pair protocol in this study. In
particular, 645 remains a geometry candidate only; 556 and 640 remain
unregistered. `TRUSTED_PAIR_PASS=0`.

## Reusable implementation and review materials

The reusable D1 diagnostic is
[`validate_alignment_g3d.py`](../../tools/exp13_cloud/g3/validate_alignment_g3d.py).
It is read-only on inputs and writes an aggregate JSON only to an explicit
output path. The private result is under
`E:\borescope_review_assets\g3d1_alignment_method_study_20261109\alignment_validation.json`.
The human atlas remains the private G3-C0 release at
`E:\borescope_review_assets\g3c0_alignment_20261108_release\index.html` and
C2 V1/V2 comparison at
`E:\borescope_review_assets\g3c2_v2_alignment_20261109_retry2\v1_v2_comparison.html`.

The D1 tool is a diagnostic around the mature C0 alignment/composite path; it
does not replace that path or rewrite prior results. Tests: D1 known-transform
control plus G3-C0 synthetic suite, 6/6 PASS. Full private run: 39/39 pairs.

## Production recommendation and Gate

Keep SIFT + fixed resize/Similarity/Affine RANSAC and the current fail-closed
candidate screen as the reusable production baseline. Add spatial holdout and
an independent feature family as diagnostic evidence, but require manual
mechanical landmark/defect-neighborhood review before geometry is accepted.
Reject on large held-out tails or method disagreement; sparse evidence remains
unverified. Avoid nonrigid warps unless a known-transform benchmark and
mechanical-shape invariance study justify them.

The evidence is not sufficient to enter the six-class 60-source paid repair
batch. Complete the ChatGPT audit Gate first; then use a small, bounded,
non-expanding review of the existing 28 C0 candidates and C2 645, with explicit
UI/Logo, local boundary, and hard/feather seam review. Revisit expansion only
after a frozen Trusted Pair protocol has produced reviewed PASS cases. No
Trusted Pair or training Gate is granted here.
