# Exp13.2-A1 Fairness Protocol and Artifact Controls

Status: `PROPOSED / NEEDS_IMPLEMENTATION_VERIFICATION`; no route is authorized or executed.

## 1. Controlled route matrix

| Route | Representation objective | Restoration pair | Polygon use | Claim supported |
|---|---|---|---|---|
| T0 | Matched-budget SimSiam only | None | 0 | Baseline for the same TRAIN budget |
| T1 | SimSiam + single-view polygon auxiliary head | No restoration pair | Same source subset and mask count as T2-T4 | Effect of single-view annotation auxiliary |
| T2 | SimSiam + ordinary two-view polygon auxiliary head | Paired views but no feature-difference localization | Same polygon budget | Effect of paired-view architecture without restoration-guided difference |
| T3 | SimSiam + multi-scale restoration-guided localization | Only frozen, manually trusted pairs | Same polygon budget | Increment attributable to restoration-guided localization relative to T2 |
| T4 | T3 + masked background consistency | Same as T3 | Same polygon budget | Increment attributable to background consistency relative to T3 |

T1-T4 must use the same independent source list, the same number of polygon annotations and the same source-level sampling rule. Variant multiplicity cannot increase the label budget or effective sample count. If the trusted set is too small, the route matrix is a design artifact only and the experiment stops.

## 2. Budget and evaluation controls

Control input resolution, batch, optimizer, scheduler, number of update steps, epoch definition, augmentation policy where applicable, random seed list, checkpoint rule and compute budget. Use the same YOLO11n-seg layers 0-10 boundary, downstream neck/head initialization, first TRAIN batch and injection hash checks used by Exp12.3/12.5. T3/T4 must add no hidden decoder capacity; projection/head parameters must be counted and matched or reported as a limitation.

Use TRAIN only for representation learning. Use a frozen VAL protocol only after Gates A-C pass. TEST remains permanently excluded from tuning, route selection, threshold choice and early stopping. Historical Exp12 A/B/C values remain reference-only and are not pooled as new route results.

## 3. Artifact-shortcut controls

The following are required design controls, not existing data:

1. **Non-defect pseudo-edit negative control.** Apply the same edit pipeline to a valid background region with no target defect, then verify that the localization head does not produce a defect-like response. Building these images requires separate authorization; A1 creates none.
2. **Non-Diffusion auxiliary controls.** Use local blur, texture transplant and photometric changes as separately labeled controls, following the Exp12.4 proxy family. They must not be presented as real restorations.
3. **Support exclusion.** Remove edit support, dilated boundaries, feather halos, gray borders and invalid crop pixels from localization targets and background loss. Audit predictions inside and outside support separately.
4. **Regional response audit.** Report response statistics over defect, valid background, edit boundary, crop padding and other-defect regions. A high boundary/background response is a shortcut signal and a stop condition.
5. **Mask-to-change correspondence.** Compare polygon support with actual image-change support, but do not use MAE/SSIM/LPIPS to declare repair success. Nonzero background MAE is a measured risk, not a positive label.
6. **Source-group accounting.** Aggregate metrics by independent source; variants are correlated and cannot be treated as independent observations.

## 4. Interpreting route comparisons

- T3 vs T0 tests the combined effect of paired restoration localization plus any added head/label supervision; it does not isolate Diffusion quality.
- T3 vs T1 tests whether restoration-guided paired differences add value beyond a single-view polygon auxiliary, subject to equal label and parameter budgets.
- T3 vs T2 tests the value of the restoration-guided feature difference over an ordinary paired polygon auxiliary, but only if decoder and compute budgets match.
- T4 vs T3 isolates the proposed background-consistency term, not restoration quality itself.

Any failed matching dimension must be listed beside the result. No comparison may be called fully fair when source count, label count, parameters, update steps or input pipeline differ.

## 5. Metrics and stopping

Record loss components, gradient norms, feature standard deviation, collapse indicators, regional shortcut responses, source-level localization metrics and downstream Mask mAP50-95 with uncertainty summaries appropriate to the final source count. The exploratory minimum of 10 independent sources, at least 5 Crack and 5 corrosion, is a proposed approval threshold only, not statistical sufficiency. If it is not met, shrink the study or stop; do not pad with variants.
