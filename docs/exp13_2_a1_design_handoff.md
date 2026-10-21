# Exp13.2-A1 Design Handoff

Date: 2026-10-21

## Delivered

The A1 method design, code reuse audit, fairness protocol and future Gate definition were prepared from the verified Exp12 code surfaces and frozen Exp13/A0 facts. The design uses a shared YOLO11n-seg layers-0..10 encoder, multi-scale taps 4/6/10, SimSiam, masked background consistency and ignore-aware local localization as proposed components.

No training logic, data loader, model inference, diffusion call, checkpoint, dataset, split, image or paper repository was changed.

## Important evidence boundaries

- A0 has 88 historical candidates and 20 source groups; no candidate has a formal PASS and no trusted pair is frozen.
- The 15 promising source labels are triage only, not training samples or a usable-pair count.
- Polygon masks are human annotation and any mask-guided objective is annotation-assisted, not strictly label-free SSL.
- The Crack 470 A0 CSV caution conflicts with the spatial report's retained fraction `1.0`; A1 records the conflict and requires coordinate verification without editing A0.
- Existing Exp12 values are historical single-seed results. They do not prove multi-seed stability or TEST generalization.

## Unresolved before implementation

1. Complete candidate-level visual review and freeze only genuinely trusted pairs.
2. Resolve crop, mask support, edit halo and non-edit-region correspondence for every selected source.
3. Build separately authorized pseudo-edit negative controls and artifact audit data.
4. Decide and pre-register equal annotation, parameter, source and compute budgets for T0-T4.
5. Approve Gate A, then Gate B smoke, before any formal representation or downstream run.

## Final state

`EXP13_2_A1_DESIGN_READY_FOR_CHATGPT_REVIEW`

```text
EXP13_2_A1_DESIGN_FROZEN = NO
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

Next permitted action is ChatGPT/user review of this design. No implementation or experiment starts automatically.
