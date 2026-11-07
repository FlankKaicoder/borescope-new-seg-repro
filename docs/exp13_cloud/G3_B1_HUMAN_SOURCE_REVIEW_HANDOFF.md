# Exp13-G3-B1 Human Source Review Handoff

## Status

`G3B1_HUMAN_SOURCE_REVIEW_READY`

The audited G3-A TRAIN export was rechecked before building the source-level
view. The source-level atlas and CSV are generated under the private asset
root; no image, professional JSON, verification package, or review output is
part of this repository.

## G3-A Gate Evidence

- Frozen Split Manifest SHA256 matched
  `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`.
- ZIP SHA256 matched its sidecar; the ZIP is 194,077,228 bytes.
- Windows report status: `PASS`; issues: 0.
- Verified TRAIN set: 668 images, 668 professional JSON files, 1,266
  instances, exact source set, and matching file hashes.
- Per-class instances: Burn 287; Crack 95; Dent 128; Material missing 184;
  Tears 45; Tip curl 27; corrosion 500.
- TEST images were not opened.

## Source Review Atlas

Private output directory: `PRIVATE_ASSET_ROOT/G3B1_SOURCE_REVIEW`.

- `index.html`: one source card per TRAIN image, with original resolution,
  all-Polygon overlay, Source ID, classes, instance count and IDs, group and
  near-duplicate identifiers, historical G1/G2 exposure, and per-Polygon
  padding/UI risk.
- `source_review.csv`: 668 source rows. `polygon_ids` lists all source
  Polygons; `selected_polygon_ids` starts empty and records only explicitly
  checked targets.
- `summary.json`: source/Polygon exact-set, per-class source and instance
  counts, and group deduplication statistics.
- Polygon detail links reuse the existing private instance-level crop
  previews. Keep the original G3-A instance-atlas directory at its existing
  relative location so those links resolve.
- The existing instance-level atlas was preserved unchanged.

Per-class distinct source counts: Burn 141; Crack 80; Dent 120; Material
missing 154; Tears 45; Tip curl 27; corrosion 195. Classes overlap within a
source, so class source counts are not additive.

Deduplication summary: 668 unique Source IDs, 632 group IDs, 18 multi-source
groups covering 54 sources. The frozen Manifest provides a known
near-duplicate group for 54 sources in 18 groups; the other 614 values are
explicitly `UNKNOWN`, not inferred from `group_id`.

The historical exposure map marks source 2 as G1F-1, source 23 as G1B/G1C/G1D/G2,
and sources 260 and 470 as G2. Other sources say
`NONE_RECORDED_IN_G1_G2_AUDIT_MAP`; that label means no call is recorded in the
current audit map, not proof of no exposure.

## Review Controls

All 668 rows start `UNREVIEWED` with zero selected Polygon IDs. The offline
page supports seven-class filtering, Source ID search, individual Polygon
target checkboxes, priority A/B/C, decisions SELECT/MAYBE/REJECT, notes,
browser-local persistence, CSV download, and importing a previously reviewed
CSV. A non-UNREVIEWED decision cannot be exported without at least one
explicitly selected Polygon ID. The page has no external dependencies.

`trusted_training_pair_status` remains `PENDING_SEPARATE_GATE` for every
source. Visual repair potential and human selection do not establish a Trusted
Training Pair or paid-generation eligibility.

Open `index.html` directly in a browser. To preserve decisions outside that
browser profile, use **Download review CSV**. To resume later, open the page
and import the saved CSV; selected Polygon IDs are restored only when they
match that source's available IDs. Editing the downloaded CSV directly is
also supported.

## Build and Validation

The source builder requires the G3-A PASS report and re-runs the TRAIN export
integrity check before reading candidate images/annotations. Example invocation
from the repository root (replace each placeholder with the existing private
path; the builder does not create or modify those inputs):

```powershell
py -3 -m tools.exp13_cloud.g3.build_source_review `
  --data-root <PRIVATE_TRAIN_EXPORT> `
  --manifest <PRIVATE_TRAIN_EXPORT>\frozen_split_manifest.csv `
  --export-manifest <PRIVATE_TRAIN_EXPORT>\export_manifest.csv `
  --verification-report <PRIVATE_VERIFY_DIR>\verify.json `
  --class-mapping <PRIVATE_TRAIN_EXPORT>\class_mapping.json `
  --instance-review-csv <PRIVATE_INSTANCE_ATLAS>\review.csv `
  --instance-atlas-root <PRIVATE_INSTANCE_ATLAS> `
  --output <NEW_PRIVATE_SOURCE_ATLAS_DIR>
```

`py -3 -m unittest tools.exp13_cloud.g3.test_g3_tools -v`: 12/12 passed.
`compileall`, `git diff --check`, generated inline JavaScript syntax, and all
3,868 generated local links passed. The source CSV contains 668 unique rows
and its 1,266 Polygon IDs equal the existing instance-level set exactly.
Node VM smoke checks exercised explicit Polygon selection/priority/decision
CSV export and importing a reviewed CSV without a browser or network.

## Git and Safety

- Start HEAD: `163b30e2a1930be21fe5b529e1f2be969f441c07` on
  `phase/exp13-cloud-repair`.
- Source-level code, regression test, and sanitized documentation are the only
  intended staged content. Private data and generated atlas files stay outside
  Git.
- `tools/exp13_cloud/g2/audit_single_result.py` remains untracked, unchanged,
  and unstaged.
- The historical publication auditor stopped at its hard-coded precommit
  HEAD requirement `3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a`; this branch is at
  `163b30e2a1930be21fe5b529e1f2be969f441c07` before this change. A staged-only
  scan passed for the seven explicit paths: no credential-pattern matches,
  media/binary assets, protected script, or private outputs. This is scoped
  review only, not a full historical publication audit.
- API POST=0; GPU Training=0; TEST Image Access=0; AutoDL Startup=0; Thesis
  Repository Access=0.
- No selection, payment authorization, G3-B2 generation, or model training was
  performed. Wait for human source screening and the separate ChatGPT Gate.
