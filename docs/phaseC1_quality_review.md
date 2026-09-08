# Phase C1 Quality Review

## 1. Review purpose

This review closes the evidence-gathering portion of Phase C1 after Distribution Audit Gate C returned `CONDITIONAL_PASS`. It assesses the already generated TRAIN-only preview audit artifacts to determine whether they demonstrate a Local Change generator design defect or require a recorded manual adjudication before Gate C can be reconsidered.

This review did not run proxy training, SSL pretraining, downstream evaluation, preview generation, distribution audit, or quality-review generation. It did not modify the Local Change generator, morphology design, preview data, or any source data.

## 2. Input data

| item | value |
|---|---:|
| preview run | `phaseC1_preview_20260907T075012Z_seed42_1_0` |
| distribution-audit run | `phaseC1_distribution_audit_20260907T083235Z` |
| quality-review run | `phaseC1_quality_review_20260907T090555Z` |
| TRAIN images | 668 |
| preview pairs | 2004 |
| synthetic edits | 3991 |
| outlier records | 876 |
| unique outlier edits | 860 |
| edits with two candidate reasons | 16 |
| manual-review labels populated | 0 / 876 |

The review inputs are the four existing contact sheets, `outlier_manual_review.csv`, `family_component_analysis.csv`, `quality_review_report.json`, and the Gate C audit outputs. The preview integrity checks remained clean: all 2004 preview files and all 668 TRAIN sources were present, while VAL and TEST image access remained zero.

`quality_review_statistics.csv` is a read-only cross-tabulation of candidate type, morphology family, and source-image defect class. A row can contribute to more than one class where the source image has multiple labels; those class subtotals must therefore not be summed as edit totals. Candidate types remain separate, so the 16 two-reason edits appear once in each applicable candidate-type subtotal. All 876 `review_label` fields remain blank.

## 3. Gate C findings

Gate C found no integrity hard failure, but retained four conditional reasons:

1. `C_elongated_ribbon`: 41.74% of edits had visible-change ratio below 0.50.
2. `D_irregular_boundary`: 42.41% of edits had visible-change ratio below 0.50.
3. 96 weak-change and 780 noise-pattern candidate records were present.
4. One preview image reached image-level changed-pixel ratio 0.389822, above the frozen 0.30 conditional threshold.

The frozen Gate C definition expects at least 95% of edits to have visible-change ratio at least 0.50. The C and D values materially miss that criterion. Gate C consequently remains `CONDITIONAL_PASS`; this review does not convert it to `PASS`.

## 4. Weak-change analysis

Weak-change candidates use the frozen threshold `changed_pixel_ratio <= 0.0005`. All 96 are still unadjudicated. Their morphology distribution is 86 C, 8 D, 2 A, and 0 B. The class-associated counts are Burn 19, Crack 12, Dent 25, Material missing 19, Tears 12, Tip curl 6, and corrosion 3.

The concentration in C is evidence that the C family should remain the primary manual-review target. It does not alone prove that every such small edit is unusable: its task role can legitimately include small local changes, and no reviewer has yet labeled the individual samples.

## 5. Noise-pattern analysis

Noise-pattern candidates satisfy the frozen rule of at least 20 connected components with largest-component-ratio lower bound at most 0.20. All 780 are still unadjudicated. Their morphology distribution is 439 D, 324 C, 17 B, and 0 A. The class-associated counts are Burn 115, Crack 103, Dent 163, Material missing 184, Tears 53, Tip curl 48, and corrosion 186.

The concentration in D and C is consistent with the Gate C morphology concern. Contact-sheet inspection shows visibly fragmented masks and, in some previews, separated local changes. The sheets are suitable for triage but cannot be used as an edit-isolated judgment because each preview pair can include multiple edits while the sheet header identifies one edit.

## 6. C morphology analysis

`C_elongated_ribbon` accounts for 1011 of 3991 edits. It has 41.74% visible-change ratios below 0.50, mean connected-component count 19.84, median 7, and p95 79.5. Its 86 weak-change candidates and 324 noise-pattern candidates make it the main quality-review focus.

The observed fragmentation is inconsistent with the frozen expectation that an elongated local edit normally behave as a coherent ribbon. However, the existing evidence does not distinguish acceptable intentionally thin or curved changes from implementation artifacts on a per-edit basis. No automatic rejection or generator modification is justified during this review.

## 7. D morphology analysis

`D_irregular_boundary` accounts for 639 edits. It has 42.41% visible-change ratios below 0.50, mean connected-component count 80.23, median 43, and p95 262.3. It has 8 weak-change candidates and 439 noise-pattern candidates.

D is the dominant source of noise-pattern candidates and has substantially greater fragmentation than every other family. This is a strong quantitative risk signal, but the review labels are blank and the contact sheets are not isolated-edit displays. The evidence supports targeted manual adjudication before deciding whether D's mask/refinement behavior is a confirmed generator defect.

## 8. Connected-component analysis

| family | edits | mean components | median | p95 | fragmentation score |
|---|---:|---:|---:|---:|---:|
| A_small_low_contrast | 1146 | 1.07 | 1 | 1.0 | 0.07 |
| B_diffuse_texture | 1195 | 3.52 | 2 | 11.0 | 2.52 |
| C_elongated_ribbon | 1011 | 19.84 | 7 | 79.5 | 18.84 |
| D_irregular_boundary | 639 | 80.23 | 43 | 262.3 | 79.23 |

The reported `mean_largest_component_ratio` is a lower-bound proxy calculated as refined-mask area divided by component count. It is not a measured largest-component area and is interpreted only as the audit's candidate-screening quantity. The component counts provide the stronger evidence: fragmentation is localized to C and especially D, rather than being a global preview failure.

## 9. Local Change design-issue decision

**Decision: `INSUFFICIENT_EVIDENCE_NEEDS_MANUAL_REVIEW`.**

The quantitative results establish a nonconforming distribution risk for C and D. They do not confirm a Local Change generator design defect because no reviewer has assigned a `review_label` to any of the 860 unique candidate edits, and the available contact sheets do not isolate the named edit from other edits in the same preview. This conclusion preserves the frozen generator and morphology design.

## 10. Next-step recommendation

1. Record a human disposition for each of the 860 unique candidate edits, retaining both reasons for the 16 overlaps and leaving original audit rows unchanged.
2. Reassess the existing Gate C evidence after that annotation. Do not regenerate previews or alter thresholds merely to clear the gate.
3. If the manual dispositions confirm systematic unacceptable C/D artifacts, authorize a separately numbered generator-repair design and a new preview run; retain all existing results.
4. If they instead show that candidates are acceptable morphology-consistent changes or metric false positives, document that evidence and reconsider Gate C without changing the frozen preview artifacts.

Proxy Training Gate is not authorized. Gate C may not be promoted to `PASS` or rerun until the required manual review and a subsequent audit decision are complete.
