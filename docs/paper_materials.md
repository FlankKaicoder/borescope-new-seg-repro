# Paper / thesis materials

## Traceable tables

- Dataset evidence: `results/project_review/dataset_summary_for_report.csv`
- Final TEST main table: `results/final/paper_main_table.csv`
- Per-class TEST: `results/final/paper_per_class_test.csv`
- Three-seed robustness: `results/final/paper_three_seed_table.csv`
- Hard Mining paired ablation: `results/final/paper_hard_mining_ablation.csv`
- ROI CE vs SupCon: `results/final/paper_roi_supcon_table.csv`
- Method status: `results/final/paper_method_status.csv`

## Headline result

Frozen seed44 Baseline TEST Mask mAP50-95 is 0.271621 and Box mAP50-95 is 0.293253. These TEST results are kept separate from VAL multi-seed and system/ROI diagnostics.

## Research questions


| RQ | Final status | Evidence-backed conclusion |
|---|---|---|
| RQ1 Dataset difficulty | ANSWERED | Imbalance, confidence, localization, confusion and scale coupling all contribute; causal attribution beyond evidence is avoided. |
| RQ2 Resolution | NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE | Future work only; no claim that high resolution is ineffective. |
| RQ3 Low confidence | ANSWERED / POSITIVE_DIAGNOSTIC | 91/173 FN recoverable (52.6%), but extreme low confidence causes FP explosion. |
| RQ4 Hard Mining | ANSWERED / NOT_CONFIRMED | Paired mean -0.005543 ± 0.035346; 1/3 positive seeds. |
| RQ5 ROI classifier | PARTIALLY_ANSWERED | CE shows useful ROI separability; corrosion/background remain difficult. |
| RQ6 Stage2 | ANSWERED / NEGATIVE | Reconstructed two-stage system did not beat the single-stage baseline. |
| RQ7 SupCon | ANSWERED_FOR_ROI_REPRESENTATION | ROI macro F1 +0.015894; no segmentation-improvement claim. |
| RQ8 KD | NOT_EVALUATED / SKIPPED_BY_ENGINEERING_GATE | Engineering-cost Gate; not a negative performance result. |
| RQ9 SimSiam | NOT_EVALUATED | 0/120 trainable backbone params changed; transfer PASS_REVISED; downstream not run. |
| RQ10 Old small-data limitations | PARTIALLY_ANSWERED | Scale alone is insufficient: task difficulty, method instability, decision rule and seed sensitivity matter. |


## Negative-results wording

- Hard Mining: seed42 was positive, but three-seed budget-matched verification did not reproduce the gain.
- SupCon: improved ROI representation, not segmentation.
- KD: not evaluated because of an engineering Gate.
- SimSiam: invalid reconstruction because no trainable backbone parameter changed; not a poor downstream score.
- Resolution: not formally evaluated and remains future work.

Figures are indexed in `docs/new_dataset_figure_index.md`; every reported number traces to CSV/JSON or the frozen Markdown audit.
