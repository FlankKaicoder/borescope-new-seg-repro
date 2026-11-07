# Exp13-G3-B1 Review Export Fix Handoff

## Status

`G3B1_REVIEW_EXPORT_FIX_READY`

## Decision Rules

- `SELECT` requires at least one explicitly selected Polygon ID.
- `MAYBE` permits zero or more selected Polygon IDs.
- `REJECT` always exports an empty `selected_polygon_ids`; changing a source to
  REJECT clears checked IDs and shows a message.
- `UNREVIEWED` always has an empty selected-ID list and is excluded from the
  reviewed count.
- The fixed page displays reviewed/total and SELECT/MAYBE/REJECT/UNREVIEWED
  counts, supports stage exports with unreviewed sources, CSV import/export,
  and emergency JSON backup download.

## Recovered User Review

The emergency backup was found under
`PRIVATE_ASSET_ROOT/g3b1_review_backups/g3b1_review_emergency_backup.json`.
It contains 668 unique source records: SELECT 53, MAYBE 10, REJECT 28, and
UNREVIEWED 577. The 53 SELECT rows retain 172 explicitly selected Polygon IDs.
Every source ID matches the baseline exactly; every selected ID belongs to its
source. REJECT selections were cleared in the restored CSV. The backup has no
nonempty user notes; note fields were retained as empty strings.

The original private `source_review.csv` was not modified; its SHA256 after
recovery is `f7328db812289f0d50b95e3ae7e3b3619290acc30a84604db43921ed5c08c5f4`.
The JSON backup was also left unchanged. The restored copy is
`PRIVATE_ASSET_ROOT/g3b1_review_backups/source_review_restored_20261107.csv`.

To repeat the strict conversion into a different new path:

```powershell
py -3 -m tools.exp13_cloud.g3.restore_source_review_backup `
  --backup-json <PRIVATE_ASSET_ROOT>\g3b1_review_backups\g3b1_review_emergency_backup.json `
  --baseline-csv <PRIVATE_ASSET_ROOT>\g3b1_source_review_20261107_final\source_review.csv `
  --output-csv <NEW_PRIVATE_RESTORED_CSV>
```

The converter rejects duplicate/missing/extra source IDs, invalid decisions or
priorities, duplicate or cross-source Polygon IDs, and SELECT rows without a
target. It refuses to overwrite the output path.

## HTML Files and Explicit Resume

- Original page: `PRIVATE_ASSET_ROOT/g3b1_source_review_20261107_final/index.html`.
- Byte-identical backup:
  `PRIVATE_ASSET_ROOT/g3b1_source_review_20261107_final/index.html.pre_export_fix_20261107.bak`.
- Repaired page:
  `PRIVATE_ASSET_ROOT/g3b1_source_review_20261107_final/index_review_fixed_v3.html`.
- Restored review CSV:
  `PRIVATE_ASSET_ROOT/g3b1_review_backups/source_review_restored_20261107.csv`.

Open `index_review_fixed_v3.html` directly in a browser. It reuses the existing
manifest-scoped storage key, but browser localStorage is not required for
recovery and is not assumed to migrate across filenames/profiles. To restore
the found review, choose the restored CSV in **Import reviewed CSV**, wait for
the imported-row confirmation, then export a fresh CSV after additional edits.
Use **Download emergency JSON** for a separate structured backup. All image
and Polygon-preview links were checked in the repaired page; 3,868 local links
resolve and there are no external dependencies.

The first repair attempt at `index_review_fixed.html` exposed an escaping bug
in the repair helper; it was retained and not used. V2 is a working intermediate.
V3 is the final validated page. The source page and all byte-identical backups
remain unchanged.

## Tests

`py -3 -m unittest tools.exp13_cloud.g3.test_g3_tools -v`: 15/15 PASS.
Coverage includes SELECT with/without a target, MAYBE without a target, REJECT
with/without historical selection, REJECT target clearing, exact JSON recovery,
CSV round-trip behavior, and default UNREVIEWED. A Node VM smoke test exercised
actual page interactions, export/import, counts, and emergency JSON download.
Generated inline JavaScript syntax passed. Python compileall and
`git diff --check` passed.

## Git and Safety

- Start HEAD: `98006ccad452feb8d22de0be520c22751488ee15` on
  `phase/exp13-cloud-repair`.
- Only repair/restore tools, synthetic tests, and sanitized documentation are
  intended for Git. Private HTML, CSV, JSON backup, imagery, and Polygon data
  remain outside Git.
- `tools/exp13_cloud/g2/audit_single_result.py` remains untracked and untouched.
- The legacy publication auditor remains pinned to historical precommit HEAD
  `3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a`; a scoped staged-only scan is
  documented separately and must not be described as the full audit.
- API POST=0; GPU Training=0; TEST Image Access=0; AutoDL Startup=0; Thesis
  Repository Access=0. No G3-B2 work was performed.
