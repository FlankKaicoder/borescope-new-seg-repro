# Exp13-G1F-2 fixed offline geometric correspondence audit

Result: **GEOMETRY_CANDIDATE**.

Stage status: `G1F_2_ALIGNMENT_AUDIT_COMPLETE / PENDING_CHATGPT_AND_USER_PAIR_GATE`.

The user considers G1F-1 pseudo-normal visual quality acceptable. That judgment
is recorded independently of this geometric screen; it does not itself approve
pixel correspondence or a Local Change training Pair. No provider/protocol
revalidation or cloud generation is performed.

## Frozen evidence and implementation

Base commit: `974b4a1d07ef0612f0b03adf6767380e03143db4`, branch
`phase/exp13-cloud-repair`; initial working tree was clean and remote matched.

| Frozen input | Dimensions | SHA256 verified before/after |
|---|---|---|
| ROI | 320x240 | `887a206bb4c550f9b8ba006684a93d1089619f06bec57c3d1e6e16648dabd74a` |
| RGBA mask, 2394 transparent pixels | 320x240 | `da23f8bb1533a83f107bf8b2ef5d5d3f072bc0a9f5ba827393b60af107845dfc` |
| Saved API raw PNG | 1024x1024 | `5554bfd54f54709fd20f325c1914b8582b083fce860c70567d26ce7ff85d4a1a` |

Code: [audit_alignment.py](../../../tools/exp13_cloud/g1f2/audit_alignment.py).
Prewritten parameters: [fixed_protocol.json](../../../tools/exp13_cloud/g1f2/fixed_protocol.json).
[Reproduction constraints](../../../tools/exp13_cloud/g1f2/README.md).

Native grayscale SIFT uses fixed defaults. BF/L2 two-neighbor matching uses
Lowe ratio 0.75, with 2-pixel spatial deduplication of orientation duplicates
and many-to-one anchors. Both diagnostics exclude the original polygon plus
8-pixel dilation. The non-UI diagnostic additionally excludes ROI y<64.
Raw-side exclusions use normalized-position projection as an exclusion prior,
not a geometric assumption. Descriptor-size clearance conservatively excludes
keypoints near these masks; it is an approximate support guard.

Similarity (`estimateAffinePartial2D`) and general Affine (`estimateAffine2D`)
are each estimated once per fixed diagnostic, using RANSAC threshold 3 pixels
in ROI coordinates, maxIters=2000, confidence=.99, refineIters=10, and seed
20261101 reset before each fit. There is no algorithm/parameter search,
nonrigid deformation, fabricated landmark, equalization or input resizing.
OpenCV 4.14.0 (headless wheel 4.14.0.94), NumPy 2.5.4; one CPU thread,
OpenCL disabled. Dependencies were installed in an isolated private folder.

Implementation exception: attempt 1 completed the fixed computation and review
views but failed serializing a NumPy boolean to JSON. Only serialization was
corrected, with the identical scientific protocol rerun in a new directory.
Both directories/markers are retained. All **33** shared matching CSV/image
files are byte-identical across attempts, and protocol hashes are identical.
There were **2 execution attempts of the same fixed scheme**, not two selected
scientific configurations. No thresholds were relaxed; the original source
snapshot, partial JSON and explicit failure note remain available privately.

## Feature and RANSAC evidence

Native feature counts: **375 ROI / 652 raw**. “Initial” below counts source
queries with two raw neighbors; “distinct” is after spatial deduplication.

| Diagnostic | Eligible ROI/raw | Initial | Lowe passed | Distinct |
|---|---:|---:|---:|---:|
| Full frame, defect excluded, UI allowed | 301 / 618 | 301 | 202 | 165 |
| Non-UI and defect/neighborhood excluded | 160 / 182 | 160 | 97 | 85 |

| Diagnostic / model | Inliers / distinct | Ratio | Median error, ROI px | P90 error, ROI px | Valid overlap |
|---|---:|---:|---:|---:|---:|
| UI allowed / Similarity | 162 / 165 | 98.18% | 0.2364 | 0.6427 | 100.0000% |
| UI allowed / Affine | 162 / 165 | 98.18% | 0.2200 | 0.6352 | 99.7943% |
| Non-UI / Similarity | 85 / 85 | 100% | 0.2559 | 0.7068 | 99.8581% |
| Non-UI / Affine | 85 / 85 | 100% | 0.2393 | 0.7164 | 99.8919% |

Errors are in-sample errors on RANSAC-selected inliers, not independent
held-out validation or statistical proof. Pixel overlap is transformed source
support, not evidence that every pixel has identical physical content.

The full-frame fit includes **63** UI inliers (of 162), so UI does contribute
strongly to that diagnostic. It cannot approve geometry. The independent
non-UI diagnostic retains 85 distinct inliers with no top-band keypoints;
its conclusion therefore does not depend solely on matching logos or icons.

Non-UI inliers span **190.73 x 142.60 ROI pixels**. Their convex hull covers
**34.86%** of the below-UI ROI. The fixed 3x2 spatial grid has four supported
cells containing **15, 43, 24 and 3** inliers, with separated centroids passing
the preset >=60-pixel rule. These correspond to upper normal metal/edge areas
and a separated lower curved boundary/metal region; grid cells are spatial
proxies, not anatomical labels. The left dark cavity interior provides little
texture evidence and is not proven pixel-for-pixel by these matches.

Only **76/2394** target pixels lie within the rasterized non-UI inlier convex
hull. Most repaired-target correspondence is an extrapolation of the global
normal-structure fit, as expected after excluding the defect/neighborhood.
The local boundary at the repair therefore requires explicit human review.

## Transform interpretation and views

Raw-to-ROI Similarity scale is **0.31264629**, rotation **-0.03142 degrees**,
translation approximately **(-0.5843, -0.4057)** ROI pixels. General Affine
singular scales are **0.31279317 / 0.31248322** (anisotropy ratio **1.00099**),
polar rotation **-0.05033 degrees**. Full matrices and each private match/error
row are saved in the private JSON/CSVs.

The two fits agree within **0.189 ROI pixels** at the four original raw-canvas
corners. The inverse mapping sends ROI corners to roughly raw x=1..1022,
y=1..767. Thus stable content is consistent with near-uniform 3.2x enlargement
and a square raw canvas extending below the original ROI field. It is not
consistent with assuming the *entire* square output must be letterboxed into
the 4:3 ROI. This geometric interpretation is supported by normal-region
features, not inferred from output dimensions alone. G1F-1's earlier letterbox
statistics remain preserved diagnostic observations; they are not the present
geometric mapping and should not be used to reject this candidate.

Both non-UI models pass the prewritten >=12 inliers, median/P90<=3px,
two separated supported cells, hull>=5%, span>=80x48, overlap>=50% and
positive-determinant screen. Both are retained; no best-looking model is
selected. Raw pixels are mapped with linear resampling, independent conservative
validity masks, RGBA alpha, and purple invalid display support. Valid overlap
is **76,691/76,800** pixels (Similarity) and **76,717/76,800** (Affine).
Invalid support is never treated as normal metal or included as a Pair target.

Private asset IDs:

- Initial failed-serialization attempt: `cloud_pilot_g1f2_alignment`.
- Completed corrected attempt: `cloud_pilot_g1f2_alignment_serialization_fix`.
- `feature_review_sheet.png`: original/mask/raw, both match sets, both models'
  RANSAC inliers/outliers; per-match CSVs record ROI/raw coordinates and errors.
- `alignment_review_sheet.png`: both mapped observations, 50% overlays,
  validity masks, corrosion/metal-edge/cavity before-vs-mapped close-ups.
- `similarity_aligned_rgba.png`, `affine_aligned_rgba.png` and separate
  `*_validity_mask.png`; original inputs/raw are copied byte-identically.
- `alignment_results.json`, `supplemental_geometry_checks.json`,
  `G1F_2_ALIGNMENT_REPORT.md` and execution-error note retain the full audit.

Preliminary inspection of mapped views supports the broad cavity contour and
unaffected curved metal boundary as a plausible candidate correspondence. The
repaired upper metal/void boundary is visibly altered where corrosion was
removed, so its physical correctness and remaining local edge continuity must
be judged by the user/ChatGPT; a good global fit cannot settle this question.
There is no automatic visual PASS, no exact cavity-interior correspondence
claim and no automatic Local Change Pair admission.

## Gate and stop

**GEOMETRY_CANDIDATE** permits human Pair review only. It does not establish
long-term stability, defect-level ground truth or training eligibility. User
appearance acceptance is retained; MECHANICAL_GEOMETRY, target-local spatial
correspondence and PAIR_USABILITY remain pending the independent Pair Gate.

This stage adds **API POST=0**, GPU=0, training=0, VAL/TEST access=0. The
cloud generation ledger stays at four historical POSTs; G1F-1's permanent
budget remains occupied. No input, raw output, G1F-1 historical report or
Exp11/Exp12 frozen conclusion is modified. No private images, masks, match
coordinates, approvals, keys, signed URLs or private maps enter Git.

`G1F_2_ALIGNMENT_AUDIT_COMPLETE / PENDING_CHATGPT_AND_USER_PAIR_GATE`.
STOP: no G2, new API call, rerun/search, SSL or downstream training.
