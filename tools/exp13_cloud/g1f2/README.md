# G1F-2 bounded geometric correspondence audit

This CPU-only stage compares one frozen TRAIN corrosion2 ROI and its one saved
cloud result. User acceptance of restored appearance is recorded separately
from mechanical correspondence. No provider request, credential access, live
module import, nonrigid mapping, parameter search, VAL/TEST or training exists
in this audit. Dependencies: OpenCV headless 4.14.0.94 (cv2 4.14.0), NumPy
2.5.4 and Pillow. OpenCV uses one thread with OpenCL disabled.

`fixed_protocol.json` is written before the single scientific execution; its
hash and the source hash are persisted before feature extraction. Both the
private output directory and marker are created exclusively; output directories
cannot be reused. Do not rerun the completed experiment or select another
configuration on the basis of the result.

SIFT uses native grayscale input without resizing/equalization. The source
descriptor is queried against raw descriptors using BF/L2 k=2 and Lowe 0.75.
Two-pixel spatial bins remove SIFT orientation duplicates and many-to-one
anchors. The original defect is dilated by 8 ROI pixels in both diagnostics;
the second diagnostic additionally excludes y<64 in the ROI. Exclusions on
the raw square are projected by normalized coordinates as a conservative
prior, **not** an alignment. A descriptor-size clearance reduces contamination
from excluded areas; it cannot prove complete descriptor-support independence.
The first diagnostic allows UI and is incapable of approving geometry.

Exactly two predefined models per diagnostic use raw-to-ROI SIFT locations:
`estimateAffinePartial2D` (similarity), `estimateAffine2D` (general affine),
RANSAC threshold 3 ROI pixels, 2000 iterations, confidence .99, refineIters=10,
seed=20261101 before each estimate. No fallback or adaptive rerun occurs. Fits
with fewer than three distinct matches are recorded as unavailable. RANSAC
inlier reprojection errors are in-sample and are not independent validation.

Only the non-UI diagnostic can pass the numeric candidate screen: >=12 distinct
inliers; median/P90 <=3 ROI pixels; >=3 inliers in each of >=2 below-UI grid
cells whose centroids are >=60 pixels apart; hull area >=5% of below-UI ROI;
span >=80x48 pixels; valid overlap >=50%; positive determinant. Grid is 3x2
over [0,320)x[64,240). These extra conservative bounds are fixed before the
run, not adjusted to obtain a PASS. Grid cells are spatial support proxies,
not anatomical segmentations. Physical boundary/cavity plausibility remains a
separate visual Gate. All numerical checks can yield only GEOMETRY_CANDIDATE.
Insufficient evidence yields GEOMETRY_UNVERIFIED, not proof of geometry failure.

An aligned observation is produced only for a non-UI model passing this full
screen. Warps use original raw bytes, linear sampling and separate validity
masks with a one-pixel raw border excluded. RGBA output alpha and purple review
background explicitly label invalid regions. No filling or compositing creates
new mechanical content. If no model passes, no warp/overlay/false close-up
correspondence is produced; native inputs and feature views are retained.

The initial local execution adds the isolated CPU dependency folder to Python's
import path and runs `audit_alignment.py --output <new-private-directory>
--run-fixed-once`. This command is documentation, not another execution grant.
All industrial inputs, feature points, matching images, matrices tied to private
coordinates, aligned images and review sheets stay private. Public results are
aggregate metrics, frozen protocol/source and a sanitized report only.

Implementation correction: the first attempt completed the fixed fits and saved
views/CSVs, then failed while serializing a NumPy boolean to JSON. The only fix
converts NumPy scalar types at JSON output. The partial initial directory and
its original code/marker remain unchanged. A separately named second directory
records the identical scientific protocol with the serialization fix; there is
no threshold change, algorithm change or parameter selection. The report and
summary disclose both execution attempts.
