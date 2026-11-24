# R5 Seed42 completed formal experiment

100 Epoch / 2100 committed effective updates / 20 VAL events / 80 native passes completed with exit0 and AMP skip0. Independent JSON/file audit: **21,168 checks, 0 conflicts, 0 missing evidence**. Scientific result Gate remains pending.

Epoch100 Last FP32 MASTER Mask AP50–95 is **0.24972101442002576**, against J1 Seed42 **0.2727438798549706**: **−2.3022865434944864 percentage points**. Original J2 remains0. EMA is0.24998586298529024. Both supplementary Best roles select Epoch100 under the frozen Box+Mask fitness/latest-tie rule.

Normal running statistics and frozen standard/fixed evaluation were used. No post-hoc Best evaluation, passes beyond the 80 scheduled, TEST access or extra seed was added. The original Epoch5 HOLD and user-approved post-observation extension to Epoch10 remain disclosed in the [full handoff](../../../EXP17_J2_R5_CONNECTED_TRAINING_HANDOFF.md).

| Evidence | Location |
|---|---|
| Independent final audit | [audit_final.json](audit_final.json) |
| Machine summary and lineage | [summary.json](summary/summary.json) |
| 100 epoch reports | [epochs.csv](summary/epochs.csv) |
| 2100 step projection | [updates.csv](summary/updates.csv) |
| Pair eligibility/exposures by class | [coverage_epoch_class.csv](summary/coverage_epoch_class.csv) |
| 40 standard and40 fixed role outputs | [validation_overall.csv](summary/validation_overall.csv) |
| Per-class standard metrics | [validation_per_class.csv](summary/validation_per_class.csv) |
| Fixed confidence counts and sizes | [counts](summary/validation_fixed_counts.csv), [scales](summary/validation_size_recall.csv) |
| 80 pass receipts | [pass_receipts.csv](summary/pass_receipts.csv) |
| Checkpoint/endpoint identities | [artifact_receipts.csv](summary/artifact_receipts.csv) |
| BN and gradient diagnostics | [BN](summary/bn_mean_gap.csv), [gradients](summary/gradient_diagnostics.csv), [weight scale](summary/focus_weight_scales.csv) |
| Matched J1 comparisons | [overall](summary/primary_J1_R5_overall.csv), [seven classes](summary/primary_J1_R5_per_class.csv), [fixed](summary/primary_J1_R5_fixed.csv), [size recall](summary/primary_J1_R5_size.csv) |
| Summary file hashes | [file_index.json](summary/file_index.json) |
| Plot source/file hashes | [plot_manifest.json](figures/plot_manifest.json) |

The16 CSVs retain raw numeric precision. Scoped Git attributes preserve the byte-identical summary and figure artifacts covered by the SHA256 manifests, including CSV line endings. Standard native Precision/Recall differ in operating-point semantics from fixed conf=.25 TP/FP/FN micro averages. Size metrics are frozen TRAIN-derived size-bin VAL recall, not COCO small/medium/large AP. Pair4 gradient ratios already include4; SupCon already includes8.

Figures are provided in PNG and PDF: [performance](figures/01_scheduled_performance.png), [loss and clip](figures/02_losses_and_clipping.png), [BN/Pair/scale](figures/03_bn_pair_and_weight_scale.png). Bands show the observed min–max of four TRAIN probes, not confidence intervals. The J1 line denotes its Epoch100 endpoint, not a J1 time series.

The full private archive contains original/new complete Checkpoints, original authorization/configuration/claim, permanent journal, saved args, native endpoint files and frozen production source. It excludes `epoch_cache` pixels and `smoke_output`, which retain their own server/history evidence. [Archive receipt](../private_archive_receipts.json) records the archive SHA and800 verified members. Large binaries, authorization JSON, industrial inputs and raw logs are excluded from Git.

The initial final summary misread historical Best selection receipts against rotating slot files; it is retained privately. The corrected summary in this directory verifies current blobs strictly and validates historical selection against immutable receipts and scheduled role identity. [Revision and tests](../final_summary_revision.json) explain the reporting-only fix. No training state, ledger or official result changed.

Offline reproduction uses `diagnostics/exp17_j2_r5/audit_final.py`, `summarize.py` and `plot_full.py` with the private archive's distinct `run`, `anchor`, `gate`, `observation` roots. Pass the observation root explicitly; the old output completion/recovery pointer intentionally remain at the historical Epoch5 HOLD. These tools do not construct models or evaluate data. The frozen production source is identical to the262 files verified at commit `cdad4e6b136af196dc812c4d86d546bf88afeffa`; final reporting changes are separate.
