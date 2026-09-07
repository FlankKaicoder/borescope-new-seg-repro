# Phase C1 Distribution Audit Gate C

## 1. Audit Purpose

This gate audited the already-generated Phase C-1 Local Change preview distribution without rerunning generation and without changing the generator. The objectives were to check morphology-family coverage, visible-change quality, seven-class source coverage, manifest integrity, and outlier behavior before deciding whether the preview distribution is acceptable for proxy-training authorization.

This audit does not perform proxy training, SSL pretraining, or downstream evaluation.

## 2. Data Source

| item | value |
|---|---|
| preview run | `phaseC1_preview_20260907T075012Z_seed42_1_0` |
| preview output | `/root/autodl-tmp/borescope-new-seg-repro/results/phaseC1_preview_generation/phaseC1_preview_20260907T075012Z_seed42_1_0` |
| audit run | `phaseC1_distribution_audit_20260907T083235Z` |
| audit output | `results/phaseC1_distribution_audit/phaseC1_distribution_audit_20260907T083235Z` |
| audit script | `tools/ssl/phaseC1_distribution_audit.py` |
| split manifest SHA256 | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| generation manifest SHA256 | `36bbeaecd6aa742c8b3c8af2ed3946713a0fc3b0fa1324a9e153357d0595e2ec` |
| final status | `PASS`, `formal_gate=true` |

Class mapping used only TRAIN rows from the frozen split manifest and the `labels_present` field. VAL and TEST were not read. An earlier audit run, `phaseC1_distribution_audit_20260907T083006Z`, was retained but superseded after correcting the audit script's filesystem file-count and same-seed duplicate semantics.

## 3. Checks Performed

1. Four-family global and per-class morphology coverage.
2. Per-edit changed-pixel ratio, changed-pixel count, delta pixels, and mean absolute grayscale difference.
3. Connected-component count and largest-component behavior.
4. Image-level changed-region component statistics from saved preview pairs.
5. Seven-class source, preview, edit, and morphology coverage.
6. Manifest-to-filesystem integrity and source/output uniqueness.
7. Weak-change, over-strong-change, and noise-pattern outlier candidates.

The audit did not regenerate previews or modify the generator.

## 4. Results

### 4.1 Morphology Distribution

| family | edits | percentage |
|---|---:|---:|
| `A_small_low_contrast` | 1146 | 28.7146% |
| `B_diffuse_texture` | 1195 | 29.9424% |
| `C_elongated_ribbon` | 1011 | 25.3320% |
| `D_irregular_boundary` | 639 | 16.0110% |

All four families are present. Observed fractions closely follow the frozen design probabilities.

### 4.2 Visible Change Quality

Across 3991 edits, the changed-pixel ratio had:

| statistic | value |
|---|---:|
| min | 0.0000293 |
| p1 | 0.0002173 |
| p5 | 0.0007971 |
| median | 0.0085278 |
| mean | 0.0342458 |
| p95 | 0.1480627 |
| max | 0.1789258 |

The changed-pixel count ranged from 12 to 73288, with mean 14027.07. Mean absolute grayscale difference was 16.5076. At image level, the maximum changed-pixel ratio was 0.389822, exceeding the audit's 0.30 conditional threshold.

Family-specific concerns are:

| family | visible ratio < 0.50 | connected components mean |
|---|---:|---:|
| `A_small_low_contrast` | 0.35% | 1.07 |
| `B_diffuse_texture` | 0.08% | 3.52 |
| `C_elongated_ribbon` | 41.74% | 19.84 |
| `D_irregular_boundary` | 42.41% | 80.23 |

The low-visible and fragmented behavior is concentrated in `C_elongated_ribbon` and `D_irregular_boundary`.

### 4.3 Seven-Class Coverage

| class | source images | previews | edits | families covered |
|---|---:|---:|---:|---:|
| Burn | 141 | 423 | 862 | 4/4 |
| Crack | 80 | 240 | 481 | 4/4 |
| Dent | 120 | 360 | 712 | 4/4 |
| Material missing | 154 | 462 | 930 | 4/4 |
| Tears | 45 | 135 | 269 | 4/4 |
| Tip curl | 27 | 81 | 160 | 4/4 |
| corrosion | 195 | 585 | 1132 | 4/4 |

Both minority classes, `Tears` and `Tip curl`, are covered by all four morphology families. No class is restricted to a single family.

### 4.4 Manifest Integrity

The audit found:

| check | result |
|---|---|
| expected TRAIN images | 668 |
| manifest per-image records | 2004 |
| filesystem preview files | 2004 |
| unique source images | 668 |
| unique generated outputs | 2004 |
| missing source images | 0 |
| missing generated outputs | 0 |
| duplicate preview keys | 0 |
| failed per-image records | 0 |
| seed mismatch | 0 |
| unexpected class stems | 0 |
| missing class stems | 0 |

Cross-seed source repeats are expected: each of the 668 source images appears once for each of the three seeds. No same-seed source duplication was found.

## 5. Outlier Samples

The audit marked 876 outlier rows, corresponding to 860 unique edits; 16 edits triggered multiple reasons.

| outlier type | count |
|---|---:|
| `WEAK_CHANGE_CANDIDATE` | 96 |
| `NOISE_PATTERN_CANDIDATE` | 780 |

No files were deleted or changed. The highest image-level changed-pixel ratio was `seed=1`, `image_stem=356`, with ratio 0.389822, 335 connected components, and largest component ratio 0.266658.

## 6. Gate Decision

**Phase C1 Distribution Audit Gate C: CONDITIONAL_PASS**

There were no integrity hard failures. However, the following conditions must be addressed or explicitly reviewed before any proxy-training authorization:

1. `C_elongated_ribbon` has 41.74% edits below visible-change ratio 0.50.
2. `D_irregular_boundary` has 42.41% edits below visible-change ratio 0.50.
3. 860 unique edits were flagged as weak-change or noise-pattern candidates.
4. One image reached image-level changed-pixel ratio 0.389822, above the 0.30 conditional threshold.

Because the decision is not `PASS`, this does not authorize Phase C1 Proxy Training. The next permitted step is a human review of the conditional items and contact-sheet outliers.
