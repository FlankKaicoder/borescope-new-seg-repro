# Exp13-G3-C0 review UI recovery handoff

Status: `G3C0_REVIEW_UI_RECOVERY_PASS`.

The user's recovered text file is a UTF-8 CSV with a BOM and four fields:
`source_id`, `visual_review_status`, `geometry_status`, and `review_notes`.
It contains 36 unique source IDs, exactly matching the G3-C0 page. All 36
records are reviewed: 31 `VISUAL_PASS`, 5 `NEEDS_RETRY`, zero `UNREVIEWED`.
The 28 `GEOMETRY_CANDIDATE` and 8 `ALIGNMENT_UNVERIFIED` values match the
frozen per-source geometry diagnostics. Six records contain notes, including
Chinese text and punctuation; the recovered notes are embedded in the new
private page. The recovered file SHA256 is
`a8db2795566d95b204e16edf7ab5759fdc122bb9ff9a5d58ad8837b6a4cc52ec`.

The new private page is `index_review_fixed_v5.html`, generated alongside the
original assets. It starts with the recovered state, prefers newer local
browser storage on later opens, and keeps Geometry read-only. Import validates
the complete unique source set, legal Visual values, and exact frozen Geometry
values before changing any card. CSV export uses UTF-8 BOM and RFC-style quoted
cells; a body-mounted temporary download link releases its Blob URL after a
delay. The Copy CSV content action provides a manually selectable textarea
fallback. Storage errors are caught and reported, leaving CSV import/export
available.

Private recovery counts and source-set verification are in
`index_review_fixed_v5_recovery_audit.json`. Original `index.html`, original
`review.csv`, recovered text file, source images, masks, geometry outputs and
diagnostics were not modified. The two earlier newly generated fixed-page
variants remain in the private directory; v5 is the final version with the
tested implementation.

Validation: Node UI harness 10/10 PASS; Python page-builder tests 3/3 PASS;
Python syntax checks PASS. The tests cover restored status and multiline
Chinese notes, autosave and refresh precedence, download link lifecycle and
delayed revocation, BOM, export/import round-trip, duplicate/missing IDs,
Geometry mismatch, clipboard fallback and blocked storage. No Playwright or
Puppeteer automation package was installed; a real browser DOM automation run
was unavailable. This limitation does not affect the deterministic Node UI
event tests.

Only the generic builder, UI script, synthetic tests and sanitized report are
published. Recovered review decisions and notes remain private. API POST=0,
GPU training=0, TEST image access=0, AutoDL startup=0. The protected untracked
G2 audit script remains untouched and unstaged.
