# Exp13-G3-D0/D1 handoff

Status: `G3D1_METHOD_STUDY_COMPLETE_PENDING_CHATGPT_AUDIT_GATE`.

G3-D2 correction: the holdout diagnostics below concern newly fitted D1 fold
models, not the saved C0/C2 transforms. D1 persisted full-match refit matrices,
not those fold matrices. The 15/12/1 cross-count is not C0 true correctness.
Full cohort=39 variants/17 passes per feature family; C0-only=36/16 passes.
Coverage used clipped valid-target masks; full masks remain in the frozen
batch package. See the corrected D1 report and G3-D2 independent benchmark.

The frozen G3-C0 and G3-C2 outputs were compared CPU-only with an independent
spatially held-out correspondence evaluation. The D1 diagnostic retains the
existing C0 alignment and composite implementation as the production baseline
and tests SIFT and ORB with fixed Similarity/Affine RANSAC. A known affine
synthetic control passed at 0.078 px median corner error. C0 holdout results:
SIFT 16/36 PASS, 17 FAIL, 3 insufficient; ORB 16 PASS, 19 FAIL, 1
insufficient. These are diagnostic outcomes, not retroactive geometry status
changes.

For C2, 645 passes held-out SIFT and ORB checks and remains a geometry
candidate; 556 is insufficient under SIFT and fails ORB's tail check; 640
fails both on large P90 errors. No source is a Trusted Pair. Existing local
target coverage metrics cannot prove local structure correctness, and eight
C0 records lack persisted masks needed to recompute neighborhood coverage.

The private JSON is
`E:\borescope_review_assets\g3d1_alignment_method_study_20261109\alignment_validation.json`.
Review existing private atlases at
`E:\borescope_review_assets\g3c0_alignment_20261108_release\index.html` and
`E:\borescope_review_assets\g3c2_v2_alignment_20261109_retry2\v1_v2_comparison.html`.
The sanitized method report is [G3-D0/D1 research](G3_D0_D1_ALIGNMENT_PAIR_QUALITY_RESEARCH.md).

Validation: 6/6 D1 + G3-C0 synthetic tests (1 + 5); the final 9/9 command also
included 3 G3-C1 closeout tests. 39/39 private pair variants
processed. API POST=0, GPU/training=0, TEST access=0, raw source/mask writes=0,
Trusted Pair PASS=0. The protected untracked G2 audit script was not modified.

Stop here for the ChatGPT research audit Gate. The six-class 60-source paid
repair batch is not ready; no request, training, or pair promotion follows
from this handoff.
