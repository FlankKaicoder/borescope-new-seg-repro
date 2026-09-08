# Phase C1 Code Alignment Report

## 1. Modified files

| file | change type | note |
|---|---|---|
| `tools/ssl/phaseC1_preview_generation.py` | modified | Aligned debug/formal modes, TRAIN-only root lock, hard failure propagation, manifest and final status outputs. |
| `docs/phaseC1_code_alignment_report.md` | added | This audit report. |

No tracked file was modified. The script remains an untracked migrated asset, so `git diff` does not show its content delta; final Git status is recorded below.

## 2. Each issue resolution

| issue | fixed | verification |
|---|---|---|
| Edit failure hard gate | Yes | Source-read failures, edit-generation failures, invalid masks, preview-save failures, and metadata-save failures produce `FAILED` records and `final_status.status=FAIL`. Failed edits are no longer silently skipped. |
| `--limit` bypass prevention | Yes | Added `--mode {debug,formal}`. Formal mode rejects nonzero `--limit`; debug output sets `formal_gate=false`. Formal mode still requires exactly 668 TRAIN images. |
| TRAIN-only source root lock | Yes | Added required `--approved-source-root`. The TRAIN directory and every resolved image must be inside the approved root and outside any VAL/TEST split path component. |
| Generation manifest | Yes | Added `generation_manifest.json` with run metadata, source root, image/edit counts, per-image records, seed information, and VAL/TEST access counters. |
| Final status | Yes | Added `final_status.json` with `PASS` or `FAIL`, image count, failed edit/image/source counts, seeds, mode, hard-gate reasons, and access counters. |
| Seed duplicate check | Yes | Empty, negative, or duplicate seed lists are rejected. |
| Single-edit pixel difference | Yes | Each edit compares the image immediately before and after that edit. Records include `before_change_pixels`, `after_change_pixels`, and `delta_pixels`. |

## 3. Remaining risks

1. The script has only been syntax-checked; no preview run or functional dry run was authorized in this phase.
2. Debug mode can succeed on a subset, but its `formal_gate` field remains `false`; human review must not count it as Gate B evidence.
3. The approved source root is an explicit user-supplied boundary. If the wrong root is supplied, the stem/count gate should reject it, but the invocation must still be reviewed before a formal run.
4. Output overwrite protection applies to the resolved run directory. A new run name is still required for every run.
5. Retried-then-successful edits remain recorded in `failed_attempts.csv`; they do not count as final failed edits.

## 4. Preview Generation Readiness

READY for a separately authorized Gate B preview run, in formal mode with the frozen 668-image TRAIN pool, explicit paths, and a new output directory.
