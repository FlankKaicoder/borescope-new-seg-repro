# Exp13.2-A1-R1 Design Review Response

Date: 2026-10-23 (Asia/Shanghai)
Status: `EXP13_2_A1_R1_DESIGN_READY_FOR_CHATGPT_REVIEW`

## Review provenance and scope

The user relayed ChatGPT's A1 review as `A1_DESIGN_REVIEW = CONDITIONAL_PASS / NEEDS_SPEC_REPAIR`: code reuse and basic architecture accepted, region semantics, matched controls, budgets and artifact gates needing repair. This document responds to that review; it is not a new visual review, candidate score, expert opinion or experimental result.

Only design Markdown is changed. Existing Exp12 code/reuse audit, historical handoffs/results, A0/A0-QA records, images, masks, prompts, checkpoints, datasets and registry remain unchanged. Append-only project state/decision/index entries point to this repair without changing their historical rows.

## Response matrix

| Issue | R1 repair | Evidence location | Still unresolved |
|---|---|---|---|
| Localization positives could be removed by whole edit-support exclusion | Explicit disjoint P/N/I; repair-confirmed polygon intersection forms P; reviewed normal background forms N; boundary/halo/uncertain area takes ignore precedence. Whole E is excluded only from background consistency, not localization. | Method sections 4-5 | Candidate-level confirmation, valid normal regions and exact halo/guard radii need separate review/freeze |
| Masks, slender Crack and degenerate loss cases | One geometry for images and all masks; conservative occupancy at strides 8/16/32; primary loss on full aligned grid; per-source masked focal/Dice equations and empty-region stop logic | Method sections 4-5; Gate B | Actual loader/loss implementation and smoke are not authorized |
| T2/T3 differed architecturally | Same shared encoder, taps, absolute differences, 64-channel projections, head, P/N/I target, independent sources and steps; only aligned second-view construction differs | Fairness sections 1-2 | Content/artifact distribution and actual wall time cannot be made identical; no pure causal claim |
| Budget and small-source ambiguity | Symbolic pre-registration table, proposed fixed base/auxiliary slots, annotation exposures, decoder arithmetic, source grouping and final-step checkpoint; batch/accumulation/absolute counts remain unmeasured | Fairness section 3 | No executable budget approved; formal trusted pairs currently 0 |
| Artifact checks too qualitative | Required same-process normal pseudo-edits, non-Diffusion controls, full-grid regional metrics, independent-source unit, threshold/coverage registry, calibration/audit separation and fail/HOLD rules | Fairness section 4; Gate C | Qualified control data and numerical criteria absent; no generation or inference here |
| Inconsistent minimum and authorization language | Same proposed minimum everywhere: at least 10 independent trusted sources, Crack/corrosion each at least 5; each phase requires separate authorization | Method section 7; fairness section 3; gates | Minimum is pending user approval, not statistical sufficiency |

The difference head is unchanged in purpose: it predicts the real defect target. T2's ordinary augmentation does not repair that defect, so identical target supervision deliberately makes T2 an architectural/annotation control. A T3 gain could still reflect editing traces; the negative controls and regional checks are required for interpretation.

## Static consistency checks and evidence limits

This is a specification-level repair. Tensor/decoder dimensions are static design descriptions, not new model probes; proposed decoder count 254,721 is arithmetic based on audited module dimensions. No runtime shape, gradient, memory, restoration quality, artifact detection or downstream performance test was executed.

Completed static checks: exact eight-file whitelist; append-only history preservation in state/log/index; revised/new Markdown table, fence and link structure; current-design Gate/NO-flag consistency; git diff --check; and unchanged registry parsing (46 data records, 11 columns). The original Phase C1 untracked path/status listing remains unchanged. Checks are limited to the repaired or appended Markdown; historical formatting is preserved. No CSV is edited. The final delivery records staged-file verification and actual Git commit/push status.

Unresolved: candidate review/selection, coordinate verification (including Crack 470), polygon completeness, normal-region evidence, halo/guard radii, control-data provenance, all numerical budgets and acceptance criteria, GPU feasibility, gradient conflicts and downstream efficacy. All require later explicit scope and review.

## Current decision

This repair is ready for ChatGPT audit; it does not certify that the repair or any data/experiment Gate has passed.

```text
EXP13_2_A1_DESIGN_FROZEN = NO
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```
