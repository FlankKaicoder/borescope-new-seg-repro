# 新孔探多类别分割复现与研究

项目实验阶段已关闭（`PROJECT_COMPLETE`）。项目将 993 张原始图像与 969 份专业 polygon JSON 审计、转换并冻结为 969 张监督图像、1847 个实例、7 类的 YOLO segmentation 数据集；采用 near-duplicate group-aware split（668/154/147），跨 split 泄漏为 0。

最终方法是 **YOLO11n-seg Baseline（seed44）**。Candidate Freeze 在任何正式 TEST 指标生成前提交；随后仅对这一 checkpoint 进行了冻结评估。第一次运行因客户端 SSH 中断且未产生指标，保留证据后，经用户明确授权用完全相同的 checkpoint/evaluator/参数重试一次。没有 TEST 驱动的训练、阈值扫描、模型/seed/checkpoint 选择。

## Final TEST

| Domain | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Box | 0.662385 | 0.474927 | 0.541636 | 0.293253 |
| Mask | 0.680727 | 0.498654 | 0.582704 | 0.271621 |

固定 `conf=0.25`：TP=127、FP=101、FN=158、P=0.557018、R=0.445614、F1=0.495127。TEST 共 147 张图、285 个实例。完整材料见 [docs/final_project_summary.md](docs/final_project_summary.md)、[docs/final_ablation_report.md](docs/final_ablation_report.md) 与 [results/final](results/final)。

## Checkpoint preservation

最终权重：seed44 Baseline `best.pt`  
服务器绝对路径：`/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt`  
SHA256：`2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`  
冻结 VAL Mask mAP50-95：`0.3251567516`；单次 TEST Mask mAP50-95：`0.2716207089`。

权重文件未进入 Git。释放 AutoDL 实例前必须单独下载并校验 SHA256。
