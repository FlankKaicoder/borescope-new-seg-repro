# 新数据集最终可视化索引

本索引只引用既有图，不重新推理或生成模型结果。

| Figure name | Path | Experiment | What it demonstrates | Thesis | Resume/interview |
|---|---|---|---|---|---|
| Unpaired image contact sheet | `results/dataset_audit/exp00_4_unpaired_review/artifacts/image_without_json_contact_sheet.jpg` | Exp00.4 | 24 张无 JSON 图的审查证据；最终统一排除 | YES | YES |
| Near-duplicate conflict example | `results/dataset_audit/exp00_5_conflict_review/artifacts/contact_sheets/near_0001.jpg` | Exp00.5 | 相邻视角与专业标签差异；解释 group-aware split | YES | YES |
| Low-confidence FN recovery | `results/fast_repro/exp03_low_conf/exp03_fn_recovery_summary.png` | Exp03 | 91/173 FN 的 recoverable 分类构成 | YES | YES |
| Threshold TP/FP/FN | `results/fast_repro/exp03_low_conf/exp03_threshold_tp_fp_fn.png` | Exp03 | 降阈值的 Recall/FP trade-off | YES | YES |
| Crack multiclass vs one-class | `results/fast_repro/exp04_crack_oneclass/exp04_crack_multiclass_vs_oneclass.png` | Exp04 | One-class 无清晰收益 | YES | OPTIONAL |
| Seed42 Hard Mining metrics | `results/fast_repro/exp05_hard_mining/exp05_control_vs_hard_mining_metrics.png` | Exp05 | 单种子 preliminary positive candidate | YES_WITH_CAPTION | YES_WITH_CAVEAT |
| Seed42 Hard Mining errors | `results/fast_repro/exp05_hard_mining/exp05_control_vs_hard_mining_errors.png` | Exp05 | Control/Treatment fixed-point errors | OPTIONAL | OPTIONAL |
| ROI CE confusion matrix | `results/fast_repro/figures/exp06_roi_ce/confusion_matrix_normalized.png` | Exp06.1 | corrosion/background classification difficulty | YES | YES |
| ROI SupCon confusion matrix | `results/fast_repro/figures/exp06_supcon/confusion_matrix_normalized.png` | Exp06.2 | SupCon 后的 ROI class behavior | YES | YES |
| Stage2 main comparison | `results/fast_repro/figures/exp07_stage2/exp07_stage2_main_comparison.png` | Exp07 | Mode A/B 均未超过 YOLO fixed point | YES | YES |
| Three-seed Mask mAP | `results/final_verify/figures/exp10/exp10_mask_map_three_seeds.png` | Exp10 | 三种子 Baseline/Control/Treatment 结果 | YES | YES |
| Paired Hard Mining delta | `results/final_verify/figures/exp10/exp10_paired_hard_mining_delta.png` | Exp10 | Hard effect 仅 1/3 seeds 为正 | YES | YES |
| Mean and std metrics | `results/final_verify/figures/exp10/exp10_mean_std_main_metrics.png` | Exp10 | 均值与 seed sensitivity | YES | YES |
| FP/FN across seeds | `results/final_verify/figures/exp10/exp10_fp_fn_three_seeds.png` | Exp10 | 方法的 error trade-off | YES | OPTIONAL |
| Difficult-class deltas | `results/final_verify/figures/exp10/exp10_difficult_classes.png` | Exp10 | Burn/Crack/corrosion 效应不稳定 | YES | YES |

注意：Exp05 图必须标注为 seed42 preliminary evidence，并与 Exp10 三种子图同时出现，不能单独支持“Hard Mining 提升 3 点”。
