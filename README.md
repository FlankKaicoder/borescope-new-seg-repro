# 新孔探多类别分割复现与研究

Exp00–Exp11 复现与最终评估阶段已关闭（`PROJECT_COMPLETE`）；后续独立研究继续。当前分支已完成 Exp14 E0／E1-FULL seed42 正式训练与冻结 VAL 比较，结果解释及后续研究授权以 [当前权威状态](docs/PROJECT_STATE.md) 为准。

## 协作与项目导航

| 文件 | 职责 |
|---|---|
| [AGENTS.md](AGENTS.md) | 长期工作规则、职责、自主执行与暂停边界 |
| [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) | 当前已核验事实、授权范围及待决事项 |
| [ROADMAP.md](ROADMAP.md) | 研究路线与未来候选计划 |
| [docs/DECISION_LOG.md](docs/DECISION_LOG.md) | 重大科研决策及适用范围 |
| [docs/experiment_index.md](docs/experiment_index.md) | 实验、历史报告与结果入口 |
| [docs/latest_handoff.md](docs/latest_handoff.md) | 最近里程碑、验证证据与接续动作 |

[中文学习材料](docs/zh/README.md) 主要复盘 Exp00–Exp11；[Exp13 云端专项记录](docs/exp13_cloud/EXP13_CLOUD_INDEX.md) 保留该阶段证据。两者中的完成状态和阶段授权只适用于各自历史范围；当前任务的权限以用户明确授权及适用的已批准协议为准，并按 AGENTS 和当前状态核对执行边界。

## Exp00–Exp11 已关闭阶段

该阶段将 993 张原始图像与 969 份专业 polygon JSON 审计、转换并冻结为 969 张监督图像、1847 个实例、7 类的 YOLO segmentation 数据集；采用 near-duplicate group-aware split（668/154/147），跨 split 泄漏为 0。

该阶段最终方法是 **YOLO11n-seg Baseline（seed44）**。Candidate Freeze 在任何正式 TEST 指标生成前提交；随后仅对这一 checkpoint 进行了冻结评估。第一次运行因客户端 SSH 中断且未产生指标，保留证据后，经用户明确授权用完全相同的 checkpoint/evaluator/参数重试一次。没有 TEST 驱动的训练、阈值扫描、模型/seed/checkpoint 选择。

## Exp00–Exp11 Final TEST

| Domain | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Box | 0.662385 | 0.474927 | 0.541636 | 0.293253 |
| Mask | 0.680727 | 0.498654 | 0.582704 | 0.271621 |

固定 `conf=0.25`：TP=127、FP=101、FN=158、P=0.557018、R=0.445614、F1=0.495127。TEST 共 147 张图、285 个实例。完整材料见 [docs/final_project_summary.md](docs/final_project_summary.md)、[docs/final_ablation_report.md](docs/final_ablation_report.md) 与 [results/final](results/final)。

## Exp00–Exp11 Checkpoint preservation

最终权重：seed44 Baseline `best.pt`  
服务器绝对路径：`/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt`  
SHA256：`2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`  
冻结 VAL Mask mAP50-95：`0.3251567516`；单次 TEST Mask mAP50-95：`0.2716207089`。

权重文件未进入 Git。释放 AutoDL 实例前必须单独下载并校验 SHA256。
