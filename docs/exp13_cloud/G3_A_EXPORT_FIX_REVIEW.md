# Exp13-G3-A export and audit fixes

## Git state

- Starting HEAD: `970085a9aea4ee5a2f406cc92b8f99bc6a9c003f`.
- Branch: `phase/exp13-cloud-repair`.
- Final HEAD: filled after the targeted fix commit and remote verification.
- Protected `tools/exp13_cloud/g2/audit_single_result.py` remained unmodified, unstaged, and untracked.
- Old main worktree and thesis repository were not accessed.

## Fixes

The frozen split manifest uses the repository's existing fields:
`image_path`, `json_path`, `stem`, `split`, `group_id`, `sha256`,
`json_sha256`, `labels_present`, `instance_count`, `image_width`, and
`image_height`. It does not define a `source_id` column. The exporter preserves
the frozen CSV byte for byte and uses its existing `stem` only as the derived
`source_id` in a separate `export_manifest.csv`; no source field is invented in
the frozen document.

Before reading source pairs, export checks the frozen manifest SHA256 against
`35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`, schema,
TRAIN count, unique stems, split values, and group leakage. For every TRAIN row,
it checks the original image and JSON hashes against the frozen row, copies the
actual files, and records source and exported paths/hashes, stem-derived
`source_id`, group, labels, dimensions, and instance count. VAL/TEST rows are
used only for split/group metadata; their files are never opened.

Windows verification checks the frozen copy SHA, 668 source IDs and exact-set
mapping, both manifest identities, the 668 actual image and JSON file sets,
all export hashes, and source/export hash equality. It verifies JSON
`imagePath` basename against the source image identity, exact dimensions,
Polygon shape types/bounds, class names, and 1,266 total instances. It reports
per-class instance counts and fails closed on missing or ambiguous evidence.

The AutoDL exporter creates a ZIP containing actual copied files, the
byte-identical frozen Manifest, `export_manifest.csv`, class mapping, export
record, and verification report. A sibling `.sha256` file authenticates the
archive. No symlinks are included. Image and JSON files are copied from the
read-only source tree; no files are written back to it. Windows verification
checks the ZIP member set and compares every member's bytes to the unpacked,
independently verified files.

G2 aggregation now carries source issues to `top_level_issues`; any source
identity/hash or execution evidence error makes the top status `FAIL`. Execution
record integrity is distinct from billing (`COST_UNKNOWN`) and upstream
identity (`UPSTREAM_IDENTITY_UNKNOWN`). The three source roles, user visual
acceptance, geometry states, and `PAIR_USABILITY_PENDING` remain independent.
No trusted Pair is granted. A fresh CPU re-audit produced
`PASS_WITH_LIMITATIONS` with zero source issues; the private report is at
`E:\borescope_review_assets\cloud_pilot_g2_closeout_20261107_fixreview`.
Billing and upstream identity remain separately `COST_UNKNOWN` and
`UPSTREAM_IDENTITY_UNKNOWN` for all three sources.

Formal candidate atlas generation requires a PASS verification report, the
frozen Manifest SHA, current Export Manifest SHA, 668/1,266 counts, and a fresh
recomputation of exact-set and file integrity. It records instance-level
candidate rows and separate source-level counts, retains editable
`UNREVIEWED`/`SELECT`/`MAYBE`/`REJECT` decisions, and uses compressed previews
with lazy-loaded images. Review assets remain private.

## Regression tests

The synthetic fixture now covers the complete frozen Manifest -> TRAIN export
-> ZIP -> extraction -> Windows verification -> atlas path, plus rejection of:

- changed frozen Manifest bytes;
- changed Export Manifest content;
- missing professional JSON;
- incorrect image hash;
- wrong source set with the expected row count;
- extra file in the TRAIN image set;
- ZIP member bytes altered while its sidecar is recomputed;
- missing or failed atlas integrity report;
- one-source G2 audit error hidden by top-level aggregation.

It also confirms VAL metadata can be present while its image/JSON files remain
absent and unopened. No industrial data is used in tests.

## Commands

AutoDL no-GPU export, only after independent GitHub audit approval:

```bash
OUTPUT=/root/autodl-tmp/exp13_g3a_train_export_YYYYMMDD \
  bash scripts/exp13_g3a_export_train.sh
```

Copy the resulting ZIP and `.sha256` sidecar to the private Windows asset root.
Verify the archive checksum and unpack it to a new directory:

```powershell
$zip = 'E:\borescope_review_assets\g3a_train_export\train_export.zip'
$expected = (Get-Content "$zip.sha256").Split()[0]
$actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw 'TRAIN export archive SHA256 mismatch' }
Expand-Archive -LiteralPath E:\borescope_review_assets\g3a_train_export\train_export.zip `
  -DestinationPath E:\borescope_review_assets\g3a_train_export_unpacked
```

The sidecar is authoritative for the ZIP digest; `Get-FileHash` is shown for
manual comparison with its single digest value. Then run Windows verification:

```powershell
py -3 tools/exp13_cloud/g3/verify_train_export.py --mode verify `
  --manifest E:\borescope_review_assets\g3a_train_export_unpacked\frozen_split_manifest.csv `
  --export-manifest E:\borescope_review_assets\g3a_train_export_unpacked\export_manifest.csv `
  --source-root E:\borescope_review_assets\g3a_train_export_unpacked `
  --class-mapping E:\borescope_review_assets\g3a_train_export_unpacked\class_mapping.json `
  --archive E:\borescope_review_assets\g3a_train_export\train_export.zip `
  --output E:\borescope_review_assets\g3a_train_verification_YYYYMMDD\verify.json
```

After and only after that report says `PASS`, build the private atlas:

```powershell
py -3 tools/exp13_cloud/g3/build_candidate_review.py `
  --data-root E:\borescope_review_assets\g3a_train_export_unpacked `
  --manifest E:\borescope_review_assets\g3a_train_export_unpacked\frozen_split_manifest.csv `
  --export-manifest E:\borescope_review_assets\g3a_train_export_unpacked\export_manifest.csv `
  --verification-report E:\borescope_review_assets\g3a_train_verification_YYYYMMDD\verify.json `
  --class-mapping E:\borescope_review_assets\g3a_train_export_unpacked\class_mapping.json `
  --output E:\borescope_review_assets\g3a_candidate_review_YYYYMMDD
```

## Publication audit

The prior `tools/exp13_cloud/audit_publication.py` stopped at its hard-coded
precommit HEAD check: it requires `3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a`,
while this authorized task starts at `970085a9aea4ee5a2f406cc92b8f99bc6a9c003f`.
It also requires the historical exact `public_allowlist.txt`, which does not
contain these G3-A files. This was a scoped auditor rejection, not a content
leak finding. Historical data/results were not rewritten to satisfy it.

The replacement check is limited to this commit's actual staged blobs: exact
allowlisted new/modified paths, no media/locks/receipts/secrets, UTF-8 source
and documentation, syntax checks, historical-prefix preservation, and staged
diff review. This does not claim a full historical publication audit and does
not rescan previously published assets.

## Current Gate

Windows has no complete TRAIN export yet. AutoDL was not started. This commit
prepares only the gated no-GPU export path; after the requested independent
ChatGPT GitHub audit, the next authorized step is a single export, then Windows
verification, then private atlas generation. API POST=0, GPU Training=0, TEST
Image Access=0, AutoDL Startup=0 for this fix task.
