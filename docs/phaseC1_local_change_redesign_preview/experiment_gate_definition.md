# Phase C-1 Experiment Gate Definition

## 1. Purpose

This document defines the gates required before Phase C-2 Proxy Smoke can be authorized.

It is not an authorization to train.

## 2. Gate A: Design Freeze

### Objective

Freeze the Candidate A redesign design before implementation.

### Required artifacts

1. `phaseC1_design_spec.md`.
2. `synthetic_change_statistics_plan.md`.
3. `train_only_generation_audit_plan.md`.
4. `experiment_gate_definition.md`.

### Acceptance criteria

1. Only generator, mask, and proxy preprocessing are allowed to change.
2. Backbone, loss, downstream protocol, and architecture remain unchanged.
3. The proxy remains label-blind.
4. The design uses only TRAIN-only offline statistics.
5. No VAL or TEST data is referenced as a design input.
6. Open design questions are explicitly listed.
7. A human reviewer accepts the design boundary.

### Result

Gate A is not complete until the user or reviewer explicitly confirms the design freeze.

## 3. Gate B: Preview Generation

### Objective

Generate synthetic previews without training.

### Required setup

1. Use the frozen 668-image TRAIN pool only.
2. Use fixed generation seeds.
3. Write outputs to a new unique directory.
4. Do not overwrite any existing result.
5. Do not modify Exp12 code or datasets.

### Required outputs

1. Input path audit.
2. Generation manifest.
3. Per-edit mask statistics.
4. Per-image change statistics.
5. Pixel-difference statistics.
6. Qualitative contact sheet.
7. `val_images_seen=0`.
8. `test_images_seen=0`.
9. `test_accessed=false`.

### Acceptance criteria

1. Exactly 668 TRAIN images are accepted.
2. Every accepted path resolves to a read-only TRAIN source.
3. No VAL or TEST path is opened.
4. The generator does not read defect labels or masks.
5. Every edit has a nonzero refined mask after retry.
6. All required metadata is saved.
7. No existing experiment artifact is modified.

Gate B is a data-generation gate, not a performance gate.

## 4. Gate C: Distribution Audit

### Objective

Verify that the synthetic change distribution matches the frozen Candidate A design and is not dominated by a trivial shortcut.

### Required checks

1. All four morphology families are present.
2. All intended size ranges are represented.
3. No family silently dominates the generated sample.
4. Refined masks have plausible area ratios.
5. Elongated changes have aspect ratios clearly above 1.
6. Diffuse changes are not reduced to tiny ellipses.
7. Visible-change ratio is not near zero for most edits.
8. Unintended global change is minimal.
9. Connected-component counts are consistent with the intended morphology.
10. The synthetic distribution is compared with TRAIN-only defect references.

### Suggested non-performance thresholds

The following are audit thresholds, not performance promises:

1. At least 95% of edits have visible-change ratio >= 0.50.
2. At least 95% of edits have exactly one connected component, unless intentionally disconnected.
3. No single synthetic mask exceeds 20% of the letterboxed image area.
4. No image has total changed-area ratio above 30%.
5. All four families appear in at least 5% of generated images.
6. The elongated family reaches bbox aspect ratio >= 4 in the preview set.
7. The diffuse family includes masks with relative area >= 0.03.
8. No preview sample uses VAL or TEST data.

These thresholds may be revised only before Gate B begins, not after seeing the results.

### Human review

The reviewer should inspect:

1. Quantitative summaries.
2. Qualitative previews.
3. Failure cases.
4. Outlier images with excessive change.
5. Any accidental use of forbidden data.

## 5. Gate D: Training Authorization

### Objective

Decide whether Phase C-2 Proxy Smoke may begin.

### Requirements

1. Gate A, B, and C are complete.
2. The design remains unchanged after preview.
3. No code or protocol change is introduced silently.
4. The reviewer accepts the proxy distribution.
5. The next stage is explicitly authorized.

### What Gate D permits

Gate D permits only a proxy smoke run, not formal training.

A proxy smoke run should still enforce:

1. TRAIN-only access.
2. Backbone parameter-update audit.
3. Gradient audit.
4. Checkpoint reload audit.
5. No VAL access.
6. No TEST access.

Gate D does not authorize downstream segmentation training.

## 6. Stop Conditions

Do not enter Phase C-2 if:

1. The synthetic preview reveals unusable masks.
2. A single family dominates unexpectedly.
3. Most edits are invisible after refinement.
4. The generator accidentally reads labels, VAL, or TEST.
5. The generator changes the dataset or existing results.
6. The design boundary is no longer the same as the frozen spec.

## 7. Evidence Boundary

Passing all Phase C-1 gates means the redesign is technically ready for a proxy smoke check.

It does not mean:

1. The proxy will train successfully.
2. The backbone will update correctly.
3. The learned representation will transfer.
4. Seven-class segmentation will improve.
5. The redesigned method should be promoted to a thesis contribution.
