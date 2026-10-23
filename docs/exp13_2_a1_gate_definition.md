# Exp13.2-A1 Future Gate Definition

Status: `DESIGN_ONLY / NOT_AUTHORIZED`
Revision: Exp13.2-A1-R1, 2026-10-23 (Asia/Shanghai).

This repairs proposed Gate definitions; none of the experimental Gates below was executed or newly passed. The user-relayed A1 design review was `CONDITIONAL_PASS / NEEDS_SPEC_REPAIR`. The present result is ready for review, not a claim that the repair is approved.

## Authorization chain

```text
Design review approval != Data Quality PASS
Data Quality PASS != authorization to implement or train
Engineering Smoke PASS != authorization for formal SSL
Representation gate != authorization for downstream
Each transition requires a separate explicit user-approved scope
TEST is never reused for tuning, checkpoint selection or artifact calibration
```

The repaired method and [fairness/control protocol](exp13_2_a1_fairness_protocol.md) define prospective requirements. A0-QA source-level screening does not satisfy candidate-level review. Current formally trusted/frozen pairs = **0**.

## Gate A: Data Quality and pre-registration readiness

A future candidate-level Data Quality PASS requires all of:

- Traceable TRAIN-only provenance, reviewer identity/evidence and variant-level repair acceptance. Source-level promising labels or image differences are insufficient.
- Actual defect elimination, credible texture, no structural hallucination/residual defect, preserved geometry, acceptable changes outside the edit and no unacceptable artifacts. Do not hide failed repair using an ignore mask.
- Verified RAW/crop/aligned coordinates and retained mask fraction. Crack 470's historical warning versus retained fraction 1.0 is unresolved; verify coordinates without changing A0.
- Joint image/mask transformations; explicit P/N/I partition as specified in the method. P must survive edit-boundary/halo exclusion; the whole edit support cannot be removed from localization. Other unresolved defects and uncertain areas cannot become N.
- Thin-crack occupancy preservation and recorded diagnostics at strides 8/16/32; empty-P/empty-N/full-ignore cases cannot silently alter the formal schedule.
- Independent-source grouping and, pending user approval, **at least 10 independent trusted sources, Crack and corrosion each at least 5**. Multiple variants do not increase this count. This exploratory proposal is not statistical sufficiency.
- Annotation source disclosure: polygon-guided objectives are annotation-assisted, not strictly label-free SSL.

If pair counts or either class minimum fail, pause two-class formal study; propose a narrower scope or stop for review. Never auto-relax quality or start one-class training.

Separate readiness items before any formal training are a complete approved budget/configuration and the artifact-control manifests plus frozen numerical criteria in the fairness protocol. Same-process normal pseudo-edits, non-Diffusion controls and sufficient independent audit sources do not currently exist as qualified controls. These need separately authorized construction; a data-quality decision on repair pairs alone cannot waive them.

HARD_STOP/HOLD: unreviewed proposed variant, uncertain correspondence, failed repair, invalid region partition, all positives excluded, inadequate independent sources or missing quality evidence. Fix only in a separately approved review/data task. R1 does not freeze any pair.

## Gate B: separately authorized engineering smoke

Only after a Data Quality decision and explicit implementation/smoke authorization may future code be added or invoked. A tiny TRAIN-only smoke must check:

| Surface | Required verification |
|---|---|
| Paired transforms and masks | Identical coordinates; exhaustive disjoint P/N/I; invalid/other-defect exclusion; no mask input to encoder/head |
| Thin Crack and empty cases | P preserved at full resolution; per-scale occupancy recorded; empty-P, empty-N, all-ignore and empty-background behavior tested |
| Masked losses | Both BCE and Dice reductions use only valid support; zero ignore gradients; source-level normalization and finite denominators |
| T2/T3 identity | Same parameter/state shapes and decoder count, source/target schedule, optimizer steps and annotation exposure |
| Gradients/updates | Finite/nonzero encoder, projector, predictor and localization-head gradients on applicable slots; each parameter group updates as intended |
| State/resources | BN behavior, batch/accumulation/precision and real GPU peak memory measured; no guessed executable batch |
| Integrity | No NaN/Inf or collapse; checkpoint reload and layers-0..10 export hashes match; label leakage audit passes |

Not every loss uses every head in every slot: the inactive projector/predictor in pure localization slots is not a missing-gradient failure; active groups and expected update counts must be predeclared. Empty-background T4 or T2/T3 eligibility mismatches stop that route comparison.

Smoke verifies engineering only. It does not authorize formal SSL, establish H2/H3, or certify artifact immunity. Any smoke-discovered specification change requires review before formal budget/criterion freeze.

## Gate C: separately authorized TRAIN-only representation study and artifact audit

Formal start requires all data, source schedules, batch/accumulation, update totals, annotation exposures, optimizer/normalization settings, final-step checkpoint rule and artifact thresholds/manifests approved and frozen. No TBD values may remain. Control source groups must follow the predeclared auxiliary-training/calibration/audit separation. Base SimSiam uses all TRAIN 668, so prior unlabeled exposure to control-source originals is disclosed; do not call this a fully representation-unseen audit.

Track losses, per-term gradient norms/alignment, finite updates, feature std/embedding variance, collapse indicators, source exposure and export integrity. No VAL/TEST selection, best-loss checkpoint selection, early stopping for a favorable result or hidden extra auxiliary updates.

At the fixed final checkpoint, only under separate inference authorization, run required control families A/B and regional audit C on the locked TRAIN-only audit sources. Compare true defect core, valid background, edit boundary, represented invalid/crop pixels, pseudo-edit and generic-change responses using the frozen t/coverage/false-response/separation criteria in the fairness protocol. Unit = independent source; variants/pixels are not independent samples.

Artifact gate states:

- Missing control data, unknown thresholds, insufficient audit coverage or no authorized audit inference -> `ARTIFACT_SHORTCUT_GATE_NOT_ASSESSED / HOLD`.
- Weak true-defect response, boundary/padding domination, high normal/pseudo-edit response, lack of frozen defect/artifact separation or any predeclared rule violation -> `ARTIFACT_SHORTCUT_GATE_FAIL / HOLD`.
- A future pass requires every required predeclared check and coverage rule; it is conditional evidence against measured shortcuts, not proof of a unique Diffusion causal effect.

Do not tune thresholds on observed route outputs. A failure is recorded and stops promotion; protocol changes require a new reviewed version. A training run with missing controls cannot be promoted merely because its losses look stable.

## Gate D: separately authorized controlled downstream

Only after the preceding required decisions and a separate downstream authorization: use frozen TRAIN/VAL, same YOLO11n-seg encoder scope, identical COCO neck/head initialization and first TRAIN batch, same input/optimizer/schedule/update budget and evaluation strategy, and strict injection/checkpoint hashes. Report non-matched dimensions and measured compute explicitly.

Historical Exp12 single-seed results remain reference-only; run contemporaneous budget-matched T0-T4 comparisons if approved. Do not use TEST for tuning, route choice, thresholds, hard mining, artifact calibration or new evaluations. Historical Exp00-Exp11 TEST results remain frozen.

## R1 current state

`EXP13_2_A1_R1_DESIGN_READY_FOR_CHATGPT_REVIEW`

```text
EXP13_2_A1_DESIGN_FROZEN = NO
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

R1 permits documentation repair and archival only. Implementation, training, smoke, inference, generation, control-data construction, model download and VAL/TEST image access remain outside this authorization.
