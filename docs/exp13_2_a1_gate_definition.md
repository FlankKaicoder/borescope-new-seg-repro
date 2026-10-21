# Exp13.2-A1 Future Gate Definition

Status: `DESIGN_ONLY / NOT_AUTHORIZED`. These gates define conditions for a later approval; none was run in A1.

## Gate A: Data Quality

PASS requires candidate-level manual review for every proposed variant, explicit trusted-pair provenance, source-level grouping, verified crop coordinates and retained mask fraction, valid correspondence masks, no residual defect, no hallucinated structure, credible normal texture, acceptable non-edit-region change, and explicit annotation-derived supervision disclosure. Polygon-incomplete or ambiguous regions become ignore regions, not positive targets. The minimum exploratory proposal is 10 independent sources with both classes represented; failure means reduce scope or stop.

HARD STOP: any unreviewed variant, crop mismatch, structural hallucination, residual defect, empty/invalid correspondence, or attempt to count variants as independent sources.

## Gate B: Engineering smoke

Before formal training, use a tiny TRAIN-only run to verify input/output shapes, finite and nonzero gradients for encoder/projector/predictor/head, parameter updates, no NaN/Inf, no collapse, checkpoint reload, encoder export hash match, valid-mask behavior and no label leakage. Record memory use for the dual-view forward. A failed smoke gate forbids formal training.

## Gate C: TRAIN-only representation evaluation

Use fixed epochs and update budget, no VAL/TEST selection, no early stopping and no best-checkpoint selection. Report each loss component, gradient balance, feature variance, regional shortcut diagnostics, source-group sampling and checkpoint integrity. A dominant loss, collapse, boundary shortcut or background response outside tolerance is a stop condition.

## Gate D: controlled downstream

Only after A-C pass, compare T0-T4 with matched TRAIN data, same downstream initialization, same neck/head, same first batch, same optimizer, same update budget and frozen VAL evaluation. TEST is never used. Route selection, thresholds and model choice must be predeclared and source-level.

## Current authorization state

`EXP13_2_A1_DESIGN_READY_FOR_CHATGPT_REVIEW`

```text
EXP13_2_A1_DESIGN_FROZEN = NO
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

The A0 source-level screen is not Gate A. No formal route, data construction, pseudo-edit generation, training, inference, or downstream evaluation is authorized by this document.
