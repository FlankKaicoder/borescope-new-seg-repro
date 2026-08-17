# New dataset figure index

| Figure | Path / source | Scope |
|---|---|---|
| Dataset class distribution | `results/final/figures/dataset_class_distribution.png` | Dataset |
| Dataset size distribution | `results/final/figures/dataset_size_distribution.png` | Dataset |
| TEST confusion matrices | `results/final_test/exp11_retry1/artifacts/ultralytics/final_seed44_baseline/confusion_matrix*.png` | Frozen TEST |
| TEST Box/Mask PR/P/R/F1 curves | `results/final_test/exp11_retry1/artifacts/ultralytics/final_seed44_baseline/*curve.png` | Frozen TEST |
| TEST recall by size | `results/final/figures/baseline_test_size_recall.png` | Frozen TEST |
| TEST per-class metrics | `results/final/figures/final_test_per_class_metrics.png` | Frozen TEST |
| TEST error taxonomy | `results/final/figures/final_test_error_taxonomy.png` | Frozen fixed threshold |
| TEST qualitative grid | `results/final_test/exp11_retry1/artifacts/baseline_test_qualitative_grid.jpg` | Frozen TEST |
| 64 qualitative cases | `results/final_test/exp11_retry1/qualitative/` | Frozen TEST |
| Exp03 threshold/FN figures | `results/fast_repro/exp03_low_conf/` | VAL diagnostic |
| Exp10 three-seed/paired figures | `results/final_verify/figures/` | VAL multi-seed |

No figure was created from a new model run; summary figures use existing CSV evidence only.

## Report-learning consolidation (2026-08-17)

| Resource | Path | Scope |
|---|---|---|
| Chinese figure guide | `docs/zh/05_图表与可视化索引.md` | Exp00–Exp11 learning/report layer |
| Existing-figure audit | `results/report_consolidation/figure_audit.csv` | 289 pre-existing PNG/JPG files |
| Visualization gap matrix | `results/report_consolidation/visualization_gap_matrix.csv` | Existing / newly added / unavailable / inappropriate |
| New report-figure manifest | `results/report_visualization_retro/figure_manifest.csv` | 18 evidence-derived figures; no new inference |
| OpenCV decode audit | `results/report_visualization_retro/opencv_decode_audit.json` | 18/18 new PNG files PASS |

The consolidation preserved every historical figure. New figures were written only under `results/report_visualization_retro/` and were derived from frozen CSV/JSON/Markdown evidence or reused qualitative images. No training, inference, threshold selection, checkpoint selection, or TEST access was performed.
