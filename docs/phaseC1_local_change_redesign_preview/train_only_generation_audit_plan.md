# Train-only Generation Audit Plan

## 1. Objective

This plan defines how future synthetic generation for the redesigned Local Change proxy will be prevented from touching VAL or TEST data.

The rule is simple: the proxy generator may read only the frozen 668-image TRAIN pool.

## 2. Input Source

The generator should consume the frozen TRAIN image list from the seven-class repository.

Required source:

- frozen TRAIN split manifest;
- `images/train` symlink pool;
- resolved read-only source images.

The generator should not read:

1. `images/val`.
2. `images/test`.
3. Any path containing `val` or `test` as a split component.
4. VAL or TEST JSON masks.
5. Any TEST-related result file.

## 3. Runtime Path Gate

Before generation, the script should perform these checks:

1. Resolve every image path to an absolute path.
2. Confirm the parent directory is `images/train`.
3. Confirm the resolved target exists.
4. Confirm the resolved target lies under the approved read-only source root.
5. Confirm the resolved path does not contain a `val` or `test` split component.
6. Confirm the number of accepted images is exactly 668.
7. Compare the accepted image stem set against the frozen TRAIN manifest.
8. Reject duplicates and missing stems.

If any check fails, the generator must stop immediately.

## 4. Label and Mask Boundary

The redesigned generator remains label-blind.

It must not read:

1. Defect class labels.
2. Defect polygons.
3. Defect masks.
4. Any downstream label file.

Offline TRAIN-only defect statistics may be used to freeze design ranges before implementation. After the design freeze, the generator should not require defect annotations at runtime.

## 5. Output Isolation

All preview or future proxy outputs should be written to a new, uniquely named directory.

Required rules:

1. Refuse to overwrite an existing output directory.
2. Do not modify the original dataset.
3. Do not modify Exp12 checkpoints.
4. Do not modify TEST-related files.
5. Do not delete or rename existing results.
6. Write a generation manifest with input stems, seeds, output paths, and file hashes.
7. Record `val_images_seen=0` and `test_images_seen=0`.
8. Record `test_accessed=false`.

## 6. Minimal Audit Log

Each generation run should create:

1. `run_config.json` with generator version, seed list, image size, and family probabilities.
2. `input_audit.json` with path checks and exact TRAIN set verification.
3. `generation_manifest.csv` with per-image and per-edit metadata.
4. `visibility_audit.json` with visible-change summaries.
5. `final_status.json` with `PASS` or `HARD_GATE`.

The audit log should make it possible to reconstruct the run without trusting undocumented state.

## 7. Failure Conditions

The run must be marked `HARD_GATE` if:

1. Any image path is outside `images/train`.
2. Any resolved path points to VAL or TEST.
3. The accepted image count is not 668.
4. A required TRAIN stem is missing.
5. An unexpected extra image appears.
6. A defect label or mask file is loaded by the proxy generator.
7. The output directory already exists.
8. Any intended change produces no visible refined mask after retries.
9. The generation process modifies the dataset or source repository.

## 8. Review Requirement

The generated preview and audit log should be reviewed by a human before Phase C-2.

The reviewer should confirm:

1. TRAIN-only access is enforced.
2. No VAL/TEST data appears in previews or manifests.
3. The generator does not silently read labels.
4. The synthetic families are present and plausible.
5. No existing experiment artifact was overwritten.
