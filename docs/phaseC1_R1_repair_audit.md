# Phase C1-R1 Local Change V2 Repair Audit

## Scope

This independently numbered TRAIN-only experiment tests whether morphology-aware Local Change generation improves the C/D families identified as risky by the Phase C1 Quality Review. A/B use the frozen V1 implementation embedded in the R1 script. C uses a centerline-generated curved variable-width ribbon with skeleton and elongation acceptance checks. D uses a synthetic seed region with smooth contour deformation, dilation/erosion, and an XOR boundary band with component, largest-component, and boundary-score checks.

No dataset split, A/B family, V1 preview, raw source, model, training path, SSL path, or downstream evaluation was modified or accessed. VAL and TEST access are zero. Rejected C/D candidates are retained in `rejected_candidates.csv`; only accepted candidates entered the V2 preview images.

## Runs and integrity

| Item | V1 | V2 R1 |
|---|---|---|
| Generation | `results/phaseC1_preview_generation/phaseC1_preview_20260907T075012Z_seed42_1_0` | `results/phaseC1_repair_generation/phaseC1_r1_formal_20260908T084500Z` |
| Audit | `results/phaseC1_distribution_audit/phaseC1_distribution_audit_20260907T083235Z` | `results/phaseC1_repair_generation/phaseC1_r1_audit_20260908T100000Z` |
| TRAIN images | 668 | 668 |
| Preview pairs | 2004 | 2004 |
| Synthetic edits | 3991 | 4052 |
| Generation status | PASS | PASS / formal gate |
| VAL / TEST access | 0 / 0 | 0 / 0 |

The V2 generation manifest has 668 unique source stems, 2004 unique preview outputs, no missing outputs, no failed image records, and seeds `42, 1, 0`. The generator recorded 442 rejected candidates and zero exhausted quality-filter edits.

## Morphology distribution

| Family | V1 count | V1 share | V2 count | V2 share | Change |
|---|---:|---:|---:|---:|---:|
| A_small_low_contrast | 1146 | 28.71% | 1216 | 30.01% | +70 |
| B_diffuse_texture | 1195 | 29.94% | 1225 | 30.23% | +30 |
| C (elongated ribbon -> crack-like structural evolution) | 1011 | 25.33% | 1011 | 24.95% | 0 |
| D (irregular boundary -> boundary evolution) | 639 | 16.01% | 600 | 14.81% | -39 |

The family probabilities remain approximately aligned with the configured 30/30/25/15 design. C and D count changes reflect accepted edit-count draws and D quality-filter retries, not a split or family-probability change.

## Noise and weak candidates

The frozen read-only audit reports the following candidate counts:

| Candidate type | V1 | V2 | Change |
|---|---:|---:|---:|
| WEAK_CHANGE_CANDIDATE | 96 | 2 | -94 (-97.9%) |
| NOISE_PATTERN_CANDIDATE | 780 | 13 | -767 (-98.3%) |

V2 candidates are concentrated in A/B under the audit's combined-preview screening: the 2 weak records are A, and the 13 noise records are B. C and D contribute zero audit outlier records. This audit result is still `CONDITIONAL_PASS` because 13 noise and 2 weak candidates remain and one image-level changed ratio is 0.349211, above the frozen 0.30 conditional threshold.

## Connected components and visible change

| Family | V1 edits | V1 mean components | V1 visible ratio < 0.5 | V2 edits | V2 mean components | V2 visible ratio < 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| A | 1146 | 1.07 | 0.35% | 1216 | 1.05 | 0.33% |
| B | 1195 | 3.52 | 0.08% | 1225 | 3.01 | 0.00% |
| C | 1011 | 19.84 | 41.74% | 1011 | 1.00 | 0.00% |
| D | 639 | 80.23 | 42.41% | 600 | 1.00 | 0.00% |
| ALL | 3991 | 19.23 | 17.49% | 4052 | 1.62 | 0.10% |

V2 C accepted masks have component count exactly 1, skeleton length at least 24, and elongation score at least 2.0. V2 C observed skeleton lengths are in the accepted generated metrics and its audit connected-component range is 1–1. V2 D accepted masks have component count exactly 1, largest-component ratio at least 0.95, and boundary score at least 0.18; its audit connected-component range is also 1–1.

## C family analysis

V1 C was the main weak/noise risk: 41.74% of its edits were below the visible-change threshold, with mean 19.84 components and 324 noise candidates plus 86 weak candidates. V2 replaces the elongated polygon ribbon with a curved centerline and variable widths, then rejects candidates that do not remain one connected elongated structure. The accepted V2 C distribution has 1011 edits, mean component count 1.00, and zero visible-ratio-below-0.5 records or audit noise/weak candidates. This is strong evidence that the C morphology repair addresses the measured fragmentation and visibility failure.

## D family analysis

V1 D was the dominant fragmentation risk: 42.41% below the visible-change threshold, mean 80.23 components, and 439 noise candidates. V2 uses a deformed seed region and boundary evolution through dilation or erosion, retaining a single main region and a nontrivial boundary band. The accepted V2 D distribution has 600 edits, mean component count 1.00, zero visible-ratio-below-0.5 records, and zero audit noise/weak candidates. The 442 rejected candidates include candidates removed before preview admission; no D edit exhausted the retry budget.

## Gate decision

The R1 generation formal gate is `PASS`. The read-only distribution audit is `CONDITIONAL_PASS` because the frozen audit still finds 13 B noise candidates, 2 A weak candidates, and one image-level changed ratio above 0.30. The C/D-specific morphology risks identified in the prior review are materially reduced, but this report does not promote the overall Phase C1 Gate C to `PASS` or authorize any training.

This experiment ends at generation and audit. Proxy Training, SSL pretraining, downstream evaluation, dataset-split changes, and further generator repair are outside this task and remain disallowed.
