# Exp13-G3-C0 batch geometry post-processing handoff

Status: `G3C0_BATCH_ALIGNMENT_READY_FOR_VISUAL_REVIEW`.

## Inputs and method

The run used the frozen 36-source expanded batch Manifest (SHA256
`2f9d0ca453bd277a61f5b8a3a4dd6e721b9218f98ef8334a00ff952b63720faa`), the
already verified G3-A TRAIN export, selected-only RGBA Masks, the G3-B2
generation review ledger, and saved raw outputs. It verified the source image,
professional JSON, Mask, and raw-output SHA256 values, source dimensions, JSON
`imagePath` basename and dimensions, and exact source-set agreement before
processing. It did not access VAL/TEST files.

The CPU implementation reuses G1F-2's native SIFT, Lowe 0.75 matching,
Similarity/Affine RANSAC and G1C's target-only composites with a fixed 3-pixel
inward feather. G1F-2's historical 320x240 ROI limits were not reused. For
near-equal aspect ratios, the proportional resize candidate had to pass
distributed non-defect SIFT correspondence checks. Other mappings were fit
with fixed Similarity/Affine RANSAC rules scaled to source dimensions. The
feature support excludes the target Polygon and a scale-relative dilation,
dark pixels, and a 2% frame border. The raw-side defect exclusion uses a
normalized-position projection of the source mask only as an exclusion prior;
it does not assert correspondence. No source-specific UI map was available, so
UI contribution remains a visual review risk. No per-source threshold search
was run.

## Results

Of 36 sources, 28 are `GEOMETRY_CANDIDATE` and 8 are
`ALIGNMENT_UNVERIFIED`. Per-source strategy, feature count, inlier count and
ratio, reprojection errors, overlap, coverage, and criteria are in the private
`geometry_review.csv` and `geometry_diagnostics.json` files.

The 28 candidates comprise 4 SIFT-validated proportional resize mappings, 5
SIFT Similarity RANSAC mappings, and 19 SIFT Affine RANSAC mappings. The eight
unverified sources have no aligned output or composite; their raw output and
clearly labeled resize-only display preview are retained. All 28 aligned
outputs, validity masks, hard composites and inward-feather composites have
source-native dimensions. Independent validation confirmed that hard and
feather composites preserve every pixel outside the valid selected target
Mask. All 36 visual states are `UNREVIEWED`; `TRUSTED_PAIR_PASS=0`.

The private review atlas has a per-source geometry diagnostics section and
shows Original, Polygon Overlay, Raw Cloud Output, Aligned Output where
supported, Hard Composite, Feather Composite, resize-only preview, and a
target-region comparison. A private CSV separates visual decision from
geometry status. Geometry candidates are only eligible for human review; they
do not establish semantic correctness, mechanical-edge continuity, or Pair
fidelity.

## Private assets

The complete run is under the user's private asset root in the new
`g3c0_alignment_20261108_release` directory. Open its `index.html` in a local
browser. Use `review.csv` to record `VISUAL_PASS`, `VISUAL_FAIL`, or
`NEEDS_RETRY`; the initial state is `UNREVIEWED`. Do not treat a visual pass or
geometry candidate as a Trusted Pair decision.

Earlier incomplete attempts are preserved in separate new directories and
were not used as the final deliverable. No historical G3-B2 asset, request
lock, source data, Prompt V1, or raw output was modified.

## Validation and safety

- Synthetic regression tests: 5/5 PASS.
- Full 36-source exact-set, input identity, output dimensions, HTML assets, and
  review CSV checks: PASS.
- Hard/feather outside-target exact-pixel preservation: PASS.
- API POST=0; GPU training=0; TEST image access=0; Trusted Pair PASS=0.
- The protected untracked G2 audit script remains unstaged and unchanged.

## Git

- Starting HEAD: `4e0f18f816acd00580c569772274c48a2f1cd749`.
- This handoff accompanies the G3-C0 tool, synthetic tests, and required
  project registry updates. The final commit and remote HEAD are reported in
  the task completion message.
- Only general-purpose code, synthetic tests, and sanitized documentation are
  published. Images, masks, JSON annotations, raw outputs, diagnostics, and
  review CSV remain private.
- The inherited publication auditor was attempted and stopped because its
  hard-coded required base is `3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a`,
  while this task starts at `4e0f18f816acd00580c569772274c48a2f1cd749`.
  A scoped review of this exact staged file set is used instead; it checks the
  staged diff, file types/modes, credentials, private asset paths, and staged
  names. It does not claim the inherited historical publication audit passed.
