# Phase C1 Recovery Audit

Audit date: 2026-09-07. This audit was read-only. The preview generator was not
run, no model was trained, no data was generated, and no server was accessed.
The only repository addition is this audit report.

## 1. Repository State

| item | value |
|---|---|
| path | `E:\github_repositories\borescope-new-seg-repro` |
| branch | `main` |
| HEAD | `7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732` |
| remote | `https://github.com/FlankKaicoder/borescope-new-seg-repro.git` |
| status | Clean tracked tree; untracked migration and audit artifacts only |

Before adding this report, untracked paths were:

- `docs/migration_audit_phaseC1.md`
- `docs/migration_execution_phaseC1.md`
- `docs/phaseB_local_change_transfer_gap_analysis.md`
- `docs/phaseC0_local_change_redesign/`
- `docs/phaseC1_local_change_redesign_preview/`
- `docs/seven_class_ssl_environment_asset_audit.md`
- `tools/ssl/phaseC1_preview_generation.py`

There were no modified or deleted tracked files. The expected server path for a
future read-only check is `/root/autodl-tmp/borescope-new-seg-repro`; its
branch, HEAD, remote, and clean status remain unverified.

## 2. Migration Asset Verification

| file | exists | readable |
|---|---|---|
| `docs/phaseC0_local_change_redesign/local_change_pipeline_audit.md` | YES | YES |
| `docs/phaseC0_local_change_redesign/phaseC0_redesign_candidates.md` | YES | YES |
| `docs/phaseC0_local_change_redesign/seven_class_vs_qikong_local_change_gap_analysis.md` | YES | YES |
| `docs/phaseB_local_change_transfer_gap_analysis.md` | YES | YES |
| `docs/seven_class_ssl_environment_asset_audit.md` | YES | YES |
| `docs/phaseC1_local_change_redesign_preview/phaseC1_design_spec.md` | YES | YES |
| `docs/phaseC1_local_change_redesign_preview/experiment_gate_definition.md` | YES | YES |
| `docs/phaseC1_local_change_redesign_preview/synthetic_change_statistics_plan.md` | YES | YES |
| `docs/phaseC1_local_change_redesign_preview/train_only_generation_audit_plan.md` | YES | YES |
| `tools/ssl/phaseC1_preview_generation.py` | YES | YES |

All ten files have nonempty content and can be read. Their current sizes match
the migration execution record. The prior migration audit and execution reports
are also present.

## 3. Phase C1 Design Audit

### TRAIN-only status

The frozen Candidate A design requires the generator to read only the frozen
668-image `images/train` pool. It forbids VAL/TEST images and masks and forbids
defect labels and masks as runtime inputs. The preview set is specified as 668
TRAIN images across three fixed seeds, producing 2,004 synthetic pairs.

The static intent is therefore TRAIN-only and label-blind.

### VAL/TEST contamination risk

The design is clear, but the current script is not yet sufficient to prove the
full runtime path gate on its own:

1. It accepts only a directory whose resolved name is `train` and whose parent
   is `images`.
2. It rejects a resolved image path containing `val` or `test` as any path
   component.
3. It compares accepted stems to the 668 TRAIN stems from the split manifest.
4. However, it does not verify that each resolved image lies under an explicitly
   approved read-only source root.
5. The `repo_root` argument is parsed but not used for this containment check.

The script reads no label, polygon, annotation, VAL image, or TEST image path.
The remaining risk is that a caller could supply an unexpected `images/train`
directory unless an approved source-root check is added before the Gate B run.

### Result-overwrite risk

The script refuses to start if the chosen run directory exists, and it creates
the run directory with `exist_ok=False`. There is no `rm`, `unlink`, rename,
move, or recursive-delete operation. Existing Exp12 results and datasets are not
targeted by the default output path.

Two residual risks remain: duplicate seed values could make later pair images
overwrite earlier files within the same new run, and there is no explicit guard
against placing `output-root` inside the TRAIN source tree. Both should be
closed before execution.

### Randomness

Generation randomness is derived per image and seed through
`random.Random(f"phaseC1A:{seed}:{stem}")`, so the intended defaults
`[42, 1, 0]` are fixed and reproducible at the Python-library level. There is no
global or wall-clock seed. The script should additionally reject duplicate or
empty seed lists.

### Scope check

The script is preview-only. It imports only NumPy and Pillow, has no model,
optimizer, checkpoint, backward, or training-loop code, and does not invoke the
existing Exp12 SSL scripts. This matches the Phase C1 objective.

## 4. Script Static Review

### Input

- Defaults: `/root/autodl-tmp/borescope-new-seg-data/v1/images/train` and the
  frozen split manifest under `/root/autodl-tmp/borescope-new-seg-repro`.
- Accepts only image files directly under the resolved `images/train` directory.
- Requires exactly 668 accepted images and exact TRAIN stem-set equality.
- Requires 640x640 output size.
- Does not read defect labels, polygons, annotations, VAL images, or TEST
  images.

### Output

- Default output root:
  `/root/autodl-tmp/borescope-new-seg-repro/results/phaseC1_preview`.
- Creates a uniquely named run directory and refuses an existing directory.
- Writes JSON summaries, CSV metrics, failed-attempt CSV, pair JPGs, contact
  sheet, and Markdown report.
- Does not create the required `final_status.json`.
- Does not create a complete `generation_manifest.csv` with per-image/per-edit
  records, source resolved paths, source/output file hashes, or generator
  version.

### Risk list

| priority | risk | evidence and consequence |
|---|---|---|
| P0 | Partial edit failures can be omitted from the final status. | `generate_pair` records a failed requested edit locally but still returns successfully if at least one edit succeeds. The main loop only counts exceptions, so an image with one successful and one failed requested edit can contribute to a final PASS. This violates the Gate B condition that every edit must have a nonzero refined mask and the HARD_GATE rule for failed edits. |
| P0 | The `--limit` debug mode can bypass the exact 668-image Gate B requirement. | A nonzero limit truncates the accepted image list. A limited run can still print `PASS` if no generated pair fails. Formal Gate B must prohibit `limit != 0`. |
| P1 | Resolved source-root containment is not enforced. | The script verifies directory naming, forbidden split components, stem count, and stem-set equality, but does not verify that each resolved source image lies under an approved read-only raw-data root. |
| P1 | Missing required audit artifacts. | There is no `final_status.json`, no complete generation manifest, no source/output file hashes, no resolved source path per edit, and no explicit generator version. |
| P1 | Duplicate or empty seed lists are not rejected. | Duplicate seeds can overwrite pair files inside the newly created run directory; an empty list would produce an empty run rather than a loud pre-run error. |
| P1 | The image-level area constraint is not enforced. | Candidate A says at most one edit per image may exceed relative area `0.05`; the script samples each edit independently and does not reject images violating that rule or the suggested 30% total-area Gate C threshold. |
| P2 | Per-edit pixel difference is cumulative. | `difference_stats` compares the current changed array with the original array, so later edits inherit differences from earlier edits rather than measuring the isolated edit. |
| P2 | Intended-area metadata is incomplete. | The script logs refined-area ratio and visible-change ratio but not the intended-mask area ratio required by the statistics plan. |
| P2 | Output/input overlap is not explicitly rejected. | The default location is safe, but a caller could point `output-root` into the TRAIN source tree. A containment check should forbid output paths inside the dataset or approved source root. |
| P2 | Gate C is not automatically complete. | The script reports distributions, but it does not implement all pre-registered size-bin, real-defect comparison, histogram, outlier, family-threshold, and human-review checks. |

### Recommendations

Before any preview run, make a separately reviewed code-alignment pass that
keeps the script preview-only and changes no experiment protocol:

1. Add an `--approved-source-root` gate and verify every resolved source image
   is contained in that read-only root.
2. Disable `--limit` for Gate B, or require an explicit non-Gate debug mode that
   cannot produce a Gate B PASS.
3. Treat every requested edit that exhausts retries as a HARD_GATE and persist
   all failures.
4. Require unique, nonempty seeds and reject duplicate seeds.
5. Write `final_status.json` and a complete generation manifest with source
   resolved paths, source/output hashes, and generator version.
6. Enforce the per-image and per-edit area limits and record intended-mask area
   ratio.
7. Compute per-edit differences against the image state immediately before that
   edit, while retaining an image-level cumulative audit.
8. Refuse output roots that overlap the approved source root, TRAIN tree, or
   existing canonical results.
9. After implementation, register the preview run with a new experiment/run ID,
   execute in a fresh unique directory, and complete Gate B/C review before any
   proxy-smoke authorization.

## 5. Readiness Decision

**PASS for entering Phase C1 Preview Generation Gate preparation.**

**BLOCKED for direct Gate B execution.**

The repository has the design, gate definitions, audit plans, migration assets,
and preview-only implementation needed to continue experimental development.
However, the static script review found Gate B alignment defects, especially
partial-failure status handling, missing source-root containment, missing final
status/manifest artifacts, and the debug-limit path. Those must be fixed and
re-reviewed before the script is run.

The next permitted step is therefore a narrowly scoped preview-generator code
alignment and review pass. No preview generation, training, server change, or
dataset modification is authorized by this audit.
