# Synthetic Change Statistics Plan

## 1. Purpose

This plan defines how to audit the redesigned synthetic local changes before any proxy training. The audit should verify whether the generated masks and pixel differences cover the intended morphology-aware design space.

This is not a performance evaluation and does not imply downstream benefit.

## 2. Preview Set

The initial preview should use the frozen 668-image TRAIN pool only.

Recommended preview size:

- 668 TRAIN images;
- 3 fixed generation seeds;
- total 2004 synthetic image pairs.

No VAL or TEST image may be included. No training is performed in this phase.

## 3. Mask-level Metrics

For each generated edit, compute:

| Metric | Definition |
|---|---|
| Intended mask area ratio | intended mask area / letterboxed image area |
| Refined mask area ratio | refined mask area / letterboxed image area |
| Visible-change ratio | refined mask area / intended mask area |
| Bbox aspect ratio | longer bbox side / shorter bbox side |
| Relative bbox width | bbox width / image width |
| Relative bbox height | bbox height / image height |
| Perimeter | contour length of refined mask |
| Compactness | `4 * pi * area / perimeter^2` |
| Connected components | number of connected components in refined mask |
| Family | synthetic morphology family |
| Size bin | small / medium / large, derived from frozen design bins |

## 4. Image-level Metrics

For each generated image pair, compute:

1. Total changed-area ratio across all edits.
2. Number of edits.
3. Number of morphology families represented.
4. Whether any single edit exceeds the intended maximum area.
5. Whether any change extends outside all intended masks.
6. Mean and maximum grayscale absolute difference.
7. Mean absolute RGB difference.
8. Optional SSIM or local structural difference, if already available in the preview tooling.

## 5. Pixel-difference Metrics

For each refined mask, compute:

1. Mean grayscale absolute difference.
2. Median grayscale absolute difference.
3. Maximum grayscale absolute difference.
4. Percentage of mask pixels with grayscale difference >= 2, >= 5, and >= 10.
5. Mean absolute RGB difference.
6. Optional per-channel difference for R/G/B.

The current qikong reference uses grayscale difference threshold 2 as a minimal visibility signal. The seven-class preview should report multiple thresholds rather than silently choosing one after seeing results.

## 6. Real-defect Comparison

Real-defect reference statistics must use only the frozen TRAIN split.

The Phase C-1 read-only TRAIN-only pooled reference is:

| Metric | q05 | q25 | q50 | q75 | q90 | q95 |
|---|---:|---:|---:|---:|---:|---:|
| Relative polygon area | 0.000486 | 0.001865 | 0.006925 | 0.028803 | 0.102614 | 0.168355 |
| Bbox aspect ratio | 1.056 | 1.223 | 1.549 | 2.196 | 3.301 | 4.206 |
| Compactness | 0.169 | 0.347 | 0.543 | 0.731 | 0.824 | 0.850 |
| Polygon vertices | 6 | 13 | 20 | 28 | 52 | 87.75 |

The preview report may also show per-class TRAIN references, but class labels must not enter the proxy generator. They are used only offline to check whether the synthetic morphology space is too narrow or too generic.

No TEST statistic may appear in the audit.

## 7. Required Audit Outputs

The future preview script should produce:

1. Per-edit CSV with all mask-level and pixel-difference metrics.
2. Per-image CSV with image-level metrics.
3. Family-wise summary table.
4. Size-bin summary table.
5. Real-defect TRAIN-only reference comparison.
6. Histograms for area ratio, aspect ratio, compactness, visible-change ratio, and pixel difference.
7. A small stratified qualitative contact sheet covering all four families and all size bins.

The contact sheet should use TRAIN images only.

## 8. Suggested Audit Questions

The audit should answer:

1. Are all four morphology families actually generated?
2. Do the generated masks cover the intended relative-area range?
3. Are elongated changes genuinely elongated, or are they still blob-like?
4. Do soft boundaries avoid rectangle/ellipse shortcuts?
5. What fraction of intended-mask pixels become visible after refinement?
6. Are there images with excessive global change?
7. Are there edits with near-zero visible pixel difference?
8. Are there unintended changes outside the intended masks?
9. Does the synthetic distribution overly favor one family or one size bin?
10. Does the synthetic mask geometry differ from TRAIN defect geometry in a way that is likely to create a proxy shortcut?

## 9. Interpretation Boundary

Passing the synthetic-statistics audit means the generator produces the intended proxy distribution.

It does **not** mean:

1. The proxy matches real defect formation.
2. The backbone will learn useful representations.
3. Downstream segmentation will improve.
4. The redesigned method is better than Exp12.

Those questions require separately authorized proxy smoke and downstream experiments.
