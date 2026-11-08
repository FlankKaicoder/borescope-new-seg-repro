# Exp13-G3-B2 Batch Generation Review

**Status:** `G3B2_BATCH_GENERATION_COMPLETE_PENDING_VISUAL_REVIEW`

## Execution

The user explicitly authorized direct execution of the complete batch without a
price-range calculation or monetary spend cap. A private authorization record
binds that approval to the expanded Manifest, review CSV, Prompt manifest,
frozen G2 protocol and exact 36-source set. The request count cap was 36.

The batch Manifest SHA256 is
`2f9d0ca453bd277a61f5b8a3a4dd6e721b9218f98ef8334a00ff952b63720faa`. Results:

| Class | Sources |
|---|---:|
| Burn | 6 |
| Crack | 6 |
| Dent | 6 |
| Material missing | 6 |
| Tears | 6 |
| corrosion | 6 |

All 36 sources returned HTTP 200 and one output image each: 36 POSTs, 36
successes, zero failures, zero skipped, zero retries. Usage fields were
returned for all sources (aggregate input 2,312 tokens, output 9,849 tokens,
total 12,161 tokens). Per-call cost was absent in all 36 responses; actual
spend remains `COST_UNKNOWN`, and no RMB amount is asserted.

The gateway used was `https://direct-api.ccccapi.cc`, model
`gpt-image-2.5-sunburst`, endpoint `/v1/images/edits`, with `n=1`, native input
frames and selected-only RGBA Masks. The public gateway page identifies itself
as an API gateway; the upstream provider's legal identity and price schedule
were not established. This batch therefore has no independently verified
price or spend ceiling.

## Outputs and Review

All request locks, request/response records, usage, source/Mask/Prompt hashes,
raw output images and terminal records are under
`${PRIVATE_ASSET_ROOT}/g3b2_batch_generation_20261108/`. The offline comparison
atlas and editable review CSV are under
`${PRIVATE_ASSET_ROOT}/g3b2_batch_review_atlas_20261108_final/` (`index.html`,
`review.csv`). The summary and output contact sheet are alongside them. The
CSV has 36 rows, all initially `UNREVIEWED`; Prompt V2 recommendation starts
`PENDING_REVIEW`.

Provider-returned output dimensions were 1024x1024 (10), 1448x1086 (13),
1519x1036 (9), 1689x931 (2), and 1691x930 (2). None match its source
dimensions. Outputs were preserved as returned and were not resized or
registered. Trusted Pair approvals remain zero; visual review does not imply
pair fidelity.

## Verification and Safety

- 36/36 permanent request locks, terminal records, HTTP status records and
  output SHA256 values verified.
- 36/36 original, Polygon Overlay and output links exist in the private atlas.
- Offline atlas JavaScript syntax passed; review CSV exact-set is 36 and all
  decisions remain `UNREVIEWED`.
- G3 regression suite: 30/30 PASS, including no-price-cap explicit
  authorization, permanent locks, no retries, and output persistence.
- The two earlier local commits were pushed before execution. Executor
  authorization changes were committed and pushed at
  `5e67cb7825d7d0333196935995efb07b797982d2`.
- Industrial images, JSON, Masks, outputs, authorization, locks and review
  data remain private and are excluded from Git.
- Protected untracked `tools/exp13_cloud/g2/audit_single_result.py` remains
  untouched and unstaged.

API POST=36; GPU training=0; TEST image access=0; AutoDL startup=0; thesis
repository access=0.

## Next Step

Open the private `index.html`, review the original/Polygon/output comparison,
and download the completed CSV after setting visual decisions and Prompt V2
recommendations. Do not begin Local Change training or treat any output as a
trusted pixel pair before separate alignment and Pair Fidelity review.
