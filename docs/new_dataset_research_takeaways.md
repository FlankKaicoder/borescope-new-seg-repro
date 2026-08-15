# New dataset research takeaways


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


The central conclusion is methodological: apparent single-seed improvements must survive matched multi-seed verification. The project found a useful confidence diagnostic and positive ROI representation signal, but neither justifies changing the frozen final segmentation method. TEST was used once for reporting and analysis, never selection.
