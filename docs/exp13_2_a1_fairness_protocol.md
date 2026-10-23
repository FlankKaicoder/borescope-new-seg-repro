# Exp13.2-A1 Fairness Protocol and Artifact Controls

Status: `PROPOSED / NEEDS_IMPLEMENTATION_VERIFICATION`
Revision: Exp13.2-A1-R1, 2026-10-23 (Asia/Shanghai). No route or control-data construction was executed or authorized.

## 1. Controlled route matrix

| Route | Base objective | Auxiliary-slot input and objective | Polygon budget |
|---|---|---|---|
| T0 | SimSiam on TRAIN 668 | SimSiam-only on the same scheduled independent-source subset; no localization labels | Zero |
| T1 | Same base SimSiam | Real image, single-view polygon localization with the same projection/head dimensions | Same as T2-T4 |
| T2 | Same base SimSiam | Real image + spatially aligned ordinary augmentation; absolute-difference localization | Same as T1/T3/T4 |
| T3 | Same base SimSiam | Real image + trusted Diffusion restoration; same absolute-difference localization | Same as T1/T2/T4 |
| T4 | Same base SimSiam | T3 plus masked background consistency | Same as T1-T3 |

All proposed routes start from the same encoder state, use the same base TRAIN draw schedule, and have the same planned encoder optimizer-step count. T0's auxiliary slots remain SimSiam, not masked pseudo-localization; this matches updates and source exposure but changes the objective. T1 processes one localization image rather than two; its compute cannot be assumed identical. Historical Exp12 A/B/C remain reference-only; a changed budget requires contemporaneous controls.

Current formally trusted/frozen repair pairs = **0**. The 15 promising source labels do not supply a training set. Any future source manifest and canonical variant choice require separate candidate-level review and freeze; no selection is performed here.

## 2. T2/T3 structural identity

| Item | Common specification |
|---|---|
| Shared Backbone | One YOLO11n layers 0-10 parameter set; same initialization and normalization policy |
| Feature taps | Layers 4/6/10, strides 8/16/32, channels 128/128/256 |
| Difference | Absolute difference of the paired features at all three scales |
| Projections | Each bias-free Conv1x1 -> 64 channels -> GroupNorm(8,64) -> SiLU |
| Alignment/fusion | Bilinear to stride-8 grid, align_corners=False, concatenate to 192 channels |
| Localization head | Conv3x3(192,128,bias=False) -> GN(8,128) -> SiLU -> Conv1x1(128,1,bias=True) |
| Target | Same real-defect polygon and reviewer-confirmed repair support from the same source record |
| Valid regions | Identical P/N/I and transformed boundary/halo restrictions; same loss and normalization |
| Resolution and geometry | 512 square; one recorded common geometry for both images and all metadata |
| Schedule | Same independent-source sequence, mask exposure, microbatch, accumulation and auxiliary steps |
| Optimization | Same optimizer, LR/scheduler, weight decay, precision, seed streams and fixed checkpoint rule |

T2's second image is an ordinary color/brightness view of Id, with no local removal, new defect or spatial displacement. Bounds must preserve defect visibility and be pre-registered. Version 1 proposes identity geometry; any later crop/flip/resize must be synchronized exactly with both images and all masks. Independent random geometric crops are prohibited.

T3 substitutes Ir for that second image. Any additional photometric processing uses a common, predeclared policy; do not quietly give T2/T3 different interpolation, compression, normalization or extra augmentation pipelines. T2 uses the same repair-confirmation-derived P/N/I even though its ordinary view still contains the defect. Thus it tests whether a matched difference head can learn the same annotation target without a repaired counterfactual; it does not assert a true repair in T2.

T2/T3 can strictly match architecture, parameter count, source IDs, polygon IDs, transformed valid targets, input resolution, planned optimizer steps and exposure schedule. They cannot make the second-view content distribution, background reconstruction, artifact texture, effective learning difficulty, gradient magnitudes, or exact GPU wall time identical. T4 also adds loss computation. Report measured time, memory, encoder forwards and FLOP estimates later; do not equalize wall time by secretly changing steps. No pair-specific eligibility mismatch may be silently accepted.

## 3. Pre-registration budget configuration

All quantities below are **future proposals**, not executed counts. Absolute update counts, batch sizes, accumulation, precision and memory feasibility are unset pending separately authorized smoke and subsequent approval. No formal run may begin with a TBD field.

Let N be the number of distinct admitted sources, U_b base updates, U_a auxiliary updates, b_b/b_a per-device source microbatches and a_b/a_a accumulation factors. Effective batches E_b=b_b*a_b and E_a=b_a*a_a count sources, not image files. First proposal: a fixed alternating base/auxiliary schedule with U_a=U_b, pending user approval; choose and freeze the absolute positive integer before any formal experiment. No number is inferred from historical epochs.

| Configuration | Proposed common rule | Pending verification / limitation |
|---|---|---|
| Base SimSiam updates | U_b identical for T0-T4; shuffled cycles of frozen TRAIN 668 with identical draws/seeds | Absolute U_b requires approval; no VAL/TEST selection |
| Auxiliary updates | U_a identical; T0 SimSiam-only, T1 polygon, T2/T3 localization, T4 adds background | Proposed U_a=U_b; no route-dependent extra steps or resampling |
| Independent-source sampling | Uniform shuffled cycles over exactly the same N source IDs; class-stratified schedule if predeclared | At least 10 trusted sources, Crack/corrosion each at least 5, pending approval |
| Batch composition | Base: E_b TRAIN sources, two ordinary views each; auxiliary T2-T4: E_a sources, two images each | T1 has one auxiliary image/source; T0 uses two ordinary views on those sources |
| Effective batch | E_b=b_b*a_b; E_a=b_a*a_a; identical across routes within each slot type | b_b, b_a, E_b, E_a = TBD / SMOKE_REQUIRED |
| Gradient accumulation | a_b/a_a fixed across routes, loss scaled consistently; no unnoticed partial final step | Accumulation factors = TBD; accumulation does not fix BN microbatch statistics |
| Encoder update budget | U_encoder=U_b+U_a; one optimizer step per completed slot after accumulation | Same LR schedule indexed by encoder steps; skipped slot cannot count as completed |
| Auxiliary/decoder updates | T1-T4: U_a; T2/T3 identical loss-validity schedule | T4 also needs nonempty valid background; T0 has no active localization decoder |
| Annotation exposure count | For T1-T4: U_a*E_a source-target presentations; same source-by-source counts | One annotation presented once per pair, not twice per image; also log valid positive/negative pixels |
| Base image exposure | U_b*E_b source draws and twice that many ordinary views | Repeated source draws are not new independent observations |
| Decoder parameter count | T2/T3/T4: 254,721 trainable parameters in the stated projections/head | Static arithmetic only; actual module count/shape must match in smoke |
| Source grouping | One future approved canonical pair/source in first version; shared source-order manifest | Multiple variants never pad N; variants/derivatives cannot cross audit groups |
| Checkpoint selection | Fixed final approved optimizer step; integrity-checked export; no best-loss/VAL checkpoint | Failures stop the run; no retroactive rescue checkpoint selection |
| Optimizer and resource record | Same optimizer, scheduler, LR, decay, precision, seeds, normalization mode, encoder forwards logged | Values/VRAM/time require authorized verification and frozen configuration |

Decoder arithmetic: projection Conv weights (128+128+256)*64=32,768; projection GN affine=384; head Conv3x3=221,184; head GN affine=256; final Conv weight+bias=129; total=254,721. This is a design count derived from the already audited Exp12-style modules, not a new model instantiation. T1 can reuse those projection/head dimensions on single-view features; T0's inactive/missing decoder and T1's single-view work are limitations for comparisons involving those routes.

Base slots update encoder/projector/predictor; T0 auxiliary slots do likewise, whereas T1-T4 auxiliary slots update encoder/localization decoder (and T4's background loss updates the encoder). Thus T0 has U_b+U_a SimSiam-head steps, versus U_b for T1-T4; this objective difference must be reported despite matched encoder steps. Slot losses are lambda_loc L_loc (plus lambda_bg L_bg for T4) on auxiliary slots and L_SimSiam on base slots; their fixed frequency defines the scheduled version of the total objective, not simultaneous evaluation of every term on every slot.

Normalize each auxiliary loss per source before source averaging. Freeze per-source annotation presentation counts and exposure-weighting rules; identical source masks with different positive-area coverage do not imply identical supervision. Record unique IDs, repeated exposures, class counts, valid pixels, transforms, encoder/head optimizer steps, and skipped slots separately.

If N<10, either class has fewer than 5 independent trusted sources, or a planned source has no valid positive/negative region, pause the two-class formal experiment. Do not fill counts with variants, reduce review standards or automatically train a one-class model. A smaller-scope design requires explicit review and authorization. The proposed minimum is exploratory, not a guarantee of statistical power; controls may require additional source-disjoint TRAIN material.

## 4. Artifact Shortcut Gate: required future data and measurements

Required control data are currently **absent**. A1/R1 neither generates them nor runs inference. Any construction, model evaluation, calibration or training needs its own authorization. These controls are gates for interpretation, not automatic training-set additions.

### A. Same-process non-defect pseudo-edits

Use independently reviewed normal TRAIN regions, with crop, mask area/shape, feathering, checkpoint, strength, prompt family, seed policy, resizing and encoding matched to the repair procedure as far as feasible. Record every mismatch, especially prompt semantics and mask geometry; do not manufacture a claim of exact procedural identity. Reviewer-check that the control did not introduce a real-looking anomaly/hallucination before using it as an artifact-only negative.

Pairs have no true defect target in the pseudo-edit region. They are held-out diagnostics, not empty-positive restoration training pairs. Compare their localization responses with real-defect cores and matched unedited backgrounds. A response that follows normal pseudo-edits rather than defects indicates an editing shortcut.

### B. Non-Diffusion auxiliary changes

Separately authorized ordinary local blur, texture and brightness perturbations on reviewed normal regions provide controls for generic image differences. Match support area and report regional pixel-change magnitude strata without selecting controls to improve model scores. Keep these distinct from T2's ordinary photometric second view. Preserve originals and reviewed geometry; label them controls, not repairs.

### C. Regional response audit

Use raw unmasked model probabilities over the full aligned output grid. Never hard-zero predictions outside E or feed E to the head; loss masks alone cannot prevent receptive-field/global shortcuts.

| Region | Measurements per source | Failure evidence |
|---|---|---|
| Defect core P | Mean response r_P, positive coverage h_P at threshold t, masked localization IoU/Dice on P union N | Weak/missing true-defect response; trivial all-zero output |
| Valid background N | Mean r_N and above-t response fraction h_N | High background activation or dependence on global reconstruction |
| Edit boundary H | Mean r_H, fraction h_H, and r_H/(r_P+eps) when P exists | Boundary-dominant ring or activations tracking mask geometry |
| Invalid/cropped region I_crop | Response on represented padding/invalid pixels and valid-side boundary band | Reliance on gray border, padding or crop edge |
| Pseudo-edit region A | Mean r_A, fraction h_A; compare with matched unedited normal and real P | Artifact-only region responds like a real defect |

There are no predictions for pixels outside the field of view; do not report responses for cropped-away RAW pixels. Other ignored regions are reported separately as uncertain, never counted as correct background. Missing/too-small regions are recorded N/A with coverage counts, not zero. Whole-source exclusion and minimum measurable-region sizes must be predeclared.

For image-space covariates report regional normalized MAE on common decoded grids, edit/mask overlap and changes outside support, plus crop/resize/encoding provenance. These statistics cannot declare restoration success. Inspect whether response follows P versus measured change support; changes in N, pseudo-edits or the boundary can expose a shortcut even when the polygon overlaps the edit.

### Statistical unit and frozen failure rules

The statistical unit is an independent source. Aggregate derivatives/variants by a predeclared within-source mean and report the worst variant too; never use pixels/variants as independent sample counts. Report source-level values, class-wise medians/ranges and all planned coverage counts, with paired T2/T3 source differences where applicable. No significance or stable-generalization claim follows from a small exploratory set.

Before formal representation training, freeze a TRAIN-only control manifest whose sources are disjoint from auxiliary optimization and whose calibration and final-audit source groups are mutually disjoint. Also freeze region definitions, source exclusions, seed list, response threshold t, minimum region sizes, minimum control-source counts, eps, and all numerical acceptance margins below. Since base SimSiam uses all TRAIN 668, control source originals may already have been seen without control labels in that branch: this is not a fully representation-unseen audit, and the limitation must be reported. A fully unseen audit would require a separately approved change to the base source budget for every route; do not silently exclude sources or claim disjointness that is impossible under the 668-image design. TRAIN-only calibration, if needed, is a separate authorized task; freeze using predeclared reference/calibration material, then lock a source-disjoint control audit set before observing T2/T3/T4 audit outputs. No VAL/TEST threshold selection and no optimization to obtain a desired T3 advantage.

| Frozen parameter | Source-level rule after authorized inference | If unset or violated |
|---|---|---|
| t, rho_min | h_P >= rho_min on each eligible defect source | Artifact gate cannot pass if true defects are not detected |
| f_N, f_H, f_crop, f_A | h_N/h_H/h_crop/h_A <= corresponding false-response cap on each measurable audit source | Flag background/boundary/crop/pseudo-edit shortcut; stop promotion |
| k_H | r_H/(r_P+eps) <= k_H on applicable paired sources | Boundary-dominated representation fails interpretation gate |
| delta_A | r_P minus class-matched pseudo-edit source aggregate >= delta_A, using frozen matching/aggregation | No demonstrated defect/artifact separation |
| delta_generic | Non-Diffusion normal-control response increase over matched unedited background <= delta_generic | Generic-change shortcut unresolved |
| n_control, min_area, missingness rule | Sufficient independent sources and measurable regions in every required control family/class | Missing evidence -> NOT_ASSESSED / HOLD, never PASS |

All thresholds are **TBD / NEEDS_APPROVAL**, not measured values or claimed universal industrial tolerances. Freeze exact values, aggregations and allowed exception counts (proposal: zero predeclared source-rule violations) before formal runs. The full run fails promotion on any required violation; investigate via a new reviewed protocol, never move thresholds after seeing results. Absence of these data or frozen numerical criteria is a blocking gate, not evidence of safety.

Paired defect versus pseudo-edit/ordinary-change responses, low boundary/background activation, and meaningful P coverage together are required. None alone proves causality; compression, smoothing and reconstruction residuals remain possible confounders. Mask exclusion cannot erase receptive-field leakage, so full-grid response audits remain mandatory.

## 5. Claims and evaluation limits

- T3 vs T0 estimates the combined annotation-assisted restoration/localization package under matched planned encoder updates; it does not isolate diffusion from supervision, heads or objective differences.
- T3 vs T1 tests paired restoration differences against single-view polygon supervision; equal annotation exposure does not erase different input/compute structure.
- T3 vs T2 holds the paired architecture, difference head, target and exposure fixed while changing the second view. It does not fully isolate a causal Diffusion benefit: content changes and edit artifacts still differ, and controls A-C are required.
- T4 vs T3 estimates the effect of the background term on the same admitted pairs and architecture, conditional on unchanged schedules and background eligibility. It is not proof of repair quality.

Only separately authorized controlled downstream runs may use the frozen TRAIN/VAL protocol, same neck/head initialization, same first-batch hash, optimizer/budget and injection integrity checks. TEST is permanently excluded from tuning, checkpoint/route choice, control calibration and all new evaluation here. Historical Exp00-Exp11 and Exp12 evidence remains frozen.

## R1 current state

`EXP13_2_A1_R1_DESIGN_READY_FOR_CHATGPT_REVIEW`

```text
EXP13_2_A1_DESIGN_FROZEN = NO
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```
