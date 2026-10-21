# Exp13.2-A0-QA — Source Visual Screening Review

Date: 2026-10-21. This archive records a source-level qualitative first-pass screen of the existing Exp13.2-A0 review sheets. It is not a new image-generation run, model-selection exercise, complete candidate-level score, or expert industrial inspection.

## Evidence boundary

ChatGPT screened the 20 user-uploaded source-level review sheets. The labels below are qualitative triage supplied by the user for archival. They do not establish restoration quality, a pass rate, statistically significant results, or a count of usable restored pairs. Source-level screening does not mean that all 88 candidates were independently reviewed. The original A0 `human_quality_review_template.csv` and all 792 judgment cells remain unchanged and blank.

## Source-level screening

| Class | Status | Sources | Count |
|---|---|---|---:|
| Crack | PROMISING / needs detailed review | 111, 146, 260, 312, 354, 356, 468, 470 | 8 |
| corrosion | PROMISING / needs detailed review | 1, 2, 6, 7, 8, 12, 27 | 7 |
| Crack | UNCERTAIN | 86, 469 | 2 |
| corrosion | REJECT / RESTRICTED | 9, 13, 23 | 3 |

“Promising” means only that these sources merit further variant-level inspection. It is not a candidate acceptance or training-pair selection. The three restricted corrosion sources retain these explicit cautions:

- corrosion 9: prior screen observed conspicuous mechanical hallucination.
- corrosion 23: prior screen observed erroneous seam/mechanical-structure generation.
- corrosion 13: the historical center crop retains only 44.1285% of the native defect mask; a full-RAW repair cannot be claimed, and edited structural realism is doubtful.

## Crack 111 variant notes

| Variant | Qualitative observation provided | Archived disposition |
|---|---|---|
| `crack_111_v1` | Original long dark crack appears to disappear; local surface may be smoothed and a small residual dark spot may remain. | `PROMISING_LOW_CONFIDENCE / REQUIRES_VARIANT_REVIEW`; no numeric score or acceptance. |
| `crack_111_v2` | Original crack is substantially changed, but new irregular brown texture may be a residual defect or generation artifact. | `NOT_PREFERRED / REQUIRES_VARIANT_REVIEW`; no numeric score or acceptance. |

The candidate ledger preserves the source screening status separately from variant review status. The two notes above are qualitative evidence only; all other variants are `PENDING_VARIANT_REVIEW`. No variant is marked PASS.

## Shortlist and review aids

The unique archive directory is `results/exp13_restoration_feasibility/qa_20261021_090000/`:

- `source_visual_screening.csv`: one row for each of 20 sources, triage status, risk, evidence source, and next action.
- `candidate_shortlist.csv`: one row for each of 88 historical candidates, including the retained-mask fraction from A0 spatial correspondence, run provenance, source-level status, variant-level state, eligibility and reason.
- `review_priority.md`: risks to examine during detailed review.
- `candidate_freeze_protocol.md`: necessary criteria for a future, separately authorized freeze; no freeze is performed here.
- `summary.json`: record counts and explicit non-authorization flags.

Candidate variants from the same source are correlated and must remain grouped. The 15 promising sources are not 15 accepted training examples. No usable-pair count or proportion can be inferred. Pixel MAE, SSIM, LPIPS, or apparent smoothness cannot establish repair success.

## State and next stage

`EXP13_2_A0_SOURCE_VISUAL_SCREENING_COMPLETE` means only that source-level first-pass screening was archived. Variant-level formal quality review remains incomplete; no trusted repair-pair set is frozen. Exp13.2-A1 method discussion may continue. Exp13.2-B training remains unauthorized.

```text
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

Polygon masks originate from human annotations. Any future method using them for masks, selection, sampling, or representation objectives must disclose annotation-derived supervision and must not be described as purely self-supervised without qualification. No generation, model download, training, VAL/TEST image access, evaluation-protocol adjustment, or paper-repository change was performed.
