# 七类缺陷自监督扩展实验 Phase A 审计

日期：2026-09-06
范围：SSH 可达性、仓库定位、历史 Local Change / SimSiam 资产只读索引。
执行边界：未运行训练、推理、预处理，未访问 TEST 数据，未修改远端或本地代码，未改动 Git 状态，未修改 SSH 配置。

## 1. SSH 状态

本机 OpenSSH 为 `OpenSSH_for_Windows_9.5p2`。`~/.ssh/config` 中只发现一个候选 Host：

| Server role | SSH alias | Status | Notes |
|---|---|---|---|
| qikong reference server | UNKNOWN | UNKNOWN | 当前 SSH config / known_hosts 中没有可确认的 qikong 服务器别名；未猜测 IP。 |
| seven-class active experiment server | UNKNOWN | UNKNOWN | 当前 SSH config / known_hosts 中没有可确认的七类服务器别名；未猜测 IP。 |
| configured local host | jetson | REACHABLE | 主机名 `nvidia-desktop`，BatchMode 成功；限深检查 `/home/nvidia` 后未发现 `qikong-ssl-det-exp` 或 `borescope-new-seg-repro`。 |

不能仅凭 `jetson` 与两个历史任务相关，因此没有把该 Host 标记为 qikong 或七类服务器。后续需要用户提供 qikong / 七类服务器的 Host alias 或授权的连接信息；本阶段未修改任何认证配置。

## 2. 气孔项目状态

当前未找到可审计的气孔服务器仓库。GitHub origin 远端可访问：

- 当前 audited server repo path：UNKNOWN
- 历史记录中的项目根路径：`/root/autodl-tmp/qikong_ssl_det_exp`
- 当前 audited server branch：UNKNOWN
- 当前 audited server HEAD：UNKNOWN
- 当前 audited working tree：UNKNOWN
- GitHub origin：`https://github.com/FlankKaicoder/qikong-ssl-det-exp.git`
- GitHub origin status：REACHABLE
- GitHub origin branch：`main`
- GitHub origin HEAD：`0b43f124991aec2eeec6ae524592be9dcf81ae12`

说明：`git ls-remote` 只证明 GitHub 远端可访问，不能证明当前服务器工作区状态。

## 3. 七类项目状态

当前未找到可审计的七类服务器仓库。GitHub origin 远端可访问：

- 当前 audited server repo path：UNKNOWN
- 历史记录中的项目根路径：`/root/autodl-tmp/borescope-new-seg-repro`
- 当前 audited server branch：UNKNOWN
- 当前 audited server HEAD：UNKNOWN
- 当前 audited working tree：UNKNOWN
- GitHub origin：`https://github.com/FlankKaicoder/borescope-new-seg-repro.git`
- GitHub origin status：REACHABLE
- GitHub origin branch：`main`
- GitHub origin HEAD：`7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732`

说明：历史 `run_config.json` 中的 `/root/autodl-tmp/...` 路径只代表历史运行位置，不代表当前服务器已确认的仓库路径。

## 4. 气孔 Local Change 可复用资产

主要来源：`experiments_material/qikong_ssl_part3_local_change.zip`、`experiments_material/qikong_ssl_part4_final_comparison.zip` 和 GitHub 参考提交 `0b43f124...`。

| Asset | Path | Exists | Notes |
|---|---|---:|---|
| Local Change 预览脚本 | `qikong_ssl_paper_assets/part3_local_change/local_change/config_and_code/08_0_preview_local_change.py` | YES | 归档代码快照存在；GitHub 参考提交 raw HEAD 返回 200。 |
| Local Change 预训练脚本 | `qikong_ssl_paper_assets/part3_local_change/local_change/config_and_code/08_1b_pretrain_yolo11_local_change_unfrozen.py` | YES | 归档代码快照存在；GitHub 参考提交 raw HEAD 返回 200。 |
| 下游迁移/训练脚本 | `qikong_ssl_paper_assets/part3_local_change/local_change/config_and_code/08_3_train_yolo11_local_change.py` | YES | 归档代码快照存在。 |
| 预训练 summary | `qikong_ssl_paper_assets/part3_local_change/local_change/pretraining/run_summary.json` | YES | 记录 741 张 image、30 epoch、seed 42、best loss 0.10087123108298882、IoU 0.824885587557584。 |
| 训练日志 | `qikong_ssl_paper_assets/part3_local_change/local_change/pretraining/training_log.csv` | YES | 归档存在。 |
| 预训练权重 | `qikong_ssl_paper_assets/part3_local_change/local_change/pretraining/weights/encoder_best.pth`、`.../local_change_best.pt` | YES | 两个权重文件均在归档中。 |
| 下游 best 权重 | `qikong_ssl_paper_assets/part3_local_change/local_change/downstream_detection/best.pt` | YES | 归档存在。 |
| 下游 summary / args / results | `qikong_ssl_paper_assets/part3_local_change/local_change/downstream_detection/run_summary.json`、`args.yaml`、`results.csv` | YES | 记录 COCO-then-Local-Change-backbone 初始化、240/240 初始迁移、trainer verification matched。 |
| backbone 参数审计 | `qikong_ssl_paper_assets/part3_local_change/local_change/audits/route_audit/local_change_backbone_audit.json` | YES | 120/120 backbone parameter tensors 均变化，relative L2 delta 约 0.032349，finite=true。 |
| 下游公平性审计 | `qikong_ssl_paper_assets/part3_local_change/local_change/audits/downstream_fairness/` | YES | 包含 JSON、LOG 和 source identity。 |
| Part4 多种子评估 | `qikong_ssl_paper_assets/part4_final_comparison/comparison/multiseed_summary/multiseed_mean_std.csv`、`multiseed_paired_deltas.csv`、`multiseed_summary.json`、`multiseed_summary.md` | YES | Local Change TEST mAP50-95 为 0.297228 ± 0.006828；3/3 seed 相对 COCO 为正。 |
| 多种子评估脚本 | `qikong_ssl_paper_assets/part4_final_comparison/comparison/scripts/evaluate_multiseed.py` | YES | 已进入 Part4 归档。 |

结论：气孔 Local Change 的实现、checkpoint、checkpoint-transfer 机制、参数审计和下游记录均可从本地归档恢复。

## 5. 七类历史 SSL 资产

主要来源：`experiments_material/borescope_full_paper_research_archive_20260823.zip` 和 GitHub 参考提交 `7d1ca22e...`。

### A. Exp09 invalid

| Asset | Path | Exists | Notes |
|---|---|---:|---|
| Exp09 总结 | `borescope_full_paper_research_archive_20260823/experiment_summaries/Exp09/README.md`、`exp09_simsiam_fast_repro.md`、`exp09_transfer_verification_repair.md`、`zh/experiments/exp09_复盘.md` | YES | 归档文档齐全。 |
| Exp09 SSL summary | `borescope_full_paper_research_archive_20260823/results_snapshot/results/fast_repro/exp09_simsiam/ssl/summary.json` | YES | 668 张 TRAIN 图、100 epoch、SGD、no collapse。 |
| 初版 transfer report | `borescope_full_paper_research_archive_20260823/results_snapshot/results/fast_repro/exp09_simsiam/transfer_report.json` | YES | 显示只加载 160/240 tensors，changed tensor 120；未证明 trainable parameter 更新。 |
| transfer repair gate | `borescope_full_paper_research_archive_20260823/results_snapshot/results/fast_repro/exp09_transfer_repair/transfer_gate_report.json` | YES | `status=REAL_TRANSFER_FAILURE`；`coco_changed_trainable_parameter_count=0`；`witness_pass=false`。 |
| 失败链可视化 | `borescope_full_paper_research_archive_20260823/figures/exp09_ssl/results/report_visualization_retro/exp09_simsiam/backbone_parameter_audit.png`、`exp09_failure_chain.png` | YES | 可用于论文/复盘说明。 |
| Exp09 代码 | GitHub `tools/training/exp09_simsiam.py` | YES | 参考提交 raw GET 返回 200。 |

结论：早期失败实验的证明链仍完整，核心证据是 0/120 trainable parameters 更新。

### B. Corrected SimSiam / Exp12-S

| Asset | Path | Exists | Notes |
|---|---|---:|---|
| 代码 | GitHub `tools/ssl/exp12_simsiam_formal.py` | YES | 参考提交 raw HEAD 返回 200。 |
| formal SSL summary | `borescope_full_paper_research_archive_20260823/results_snapshot/results/exp12_simsiam_basic/formal_20260822T071127Z/summary.json` | YES | 668 张 TRAIN 图、100 epoch、120/120 parameters changed、`status=PASS`、未访问 VAL/TEST。 |
| 参数快照 / 更新审计 | `borescope_full_paper_research_archive_20260823/results_snapshot/results/exp12_simsiam_basic/formal_20260822T071127Z/parameter_snapshot_before.jsonl`、`parameter_snapshot_after.jsonl`、`parameter_update_report.json` | YES | 可支持逐参数恢复审计。 |
| checkpoint / 导出哈希 | 同一 formal 目录 `summary.json` 中的 `final_checkpoint_sha256`、`backbone_export_sha256` | YES | 记录存在。 |
| 下游 A/B 结果 | `borescope_full_paper_research_archive_20260823/results_snapshot/results/exp12_downstream_ab/formal_20260822T090202Z/frozen_val_comparison.json` | YES | Route B 的 seed42 VAL Mask mAP50-95 为 0.3218971509452397，Route A 为 0.29898123412927496，差值 +0.02291591681596472。 |
| 下游训练 CSV / audit | 同一 formal 目录 `route_a/`、`route_b/` 下的 `summary.json`、`training_artifacts/results.csv`、`injection_audit.json`、`trainer_initialization_audit.json` | YES | 可复现下游初始化与训练记录。 |
| Exp12 文档 | `borescope_full_paper_research_archive_20260823/experiment_summaries/Exp12/README.md`、`exp12_1_formal_and_exp12_2_audit.md`、`exp12_3_downstream_ab.md` | YES | 归档文档齐全。 |

结论：corrected SimSiam 的代码、结果、参数审计和下游冻结 VAL 记录仍可恢复。该结果仅为单种子 VAL-only 探索。

### C. Local Change / Exp12-L

| Asset | Path | Exists | Notes |
|---|---|---:|---|
| 预训练代码 | GitHub `tools/ssl/exp12_local_change.py` | YES | 参考提交 raw HEAD 返回 200。 |
| 下游代码 | GitHub `tools/training/exp12_3_downstream.py` | YES | 参考提交 raw HEAD 返回 200。 |
| formal 预训练 summary | `borescope_full_paper_research_archive_20260823/results_snapshot/results/exp12_local_change/formal_20260823T064923Z/summary.json` | YES | 668 张 TRAIN 图、30 epoch、120/120 parameters changed、proxy IoU 0.5731893437343486 → 0.9522837025965085、未访问 VAL/TEST。 |
| run config | `borescope_full_paper_research_archive_20260823/results_snapshot/results/exp12_local_change/formal_20260823T064923Z/run_config.json` | YES | 记录 P3/P4/P5 tap layers、focal BCE + soft dice、AdamW、seed 42、TRAIN exact set 668。 |
| 参数快照 / 更新审计 | 同一 formal 目录 `parameter_snapshot_before.jsonl`、`parameter_snapshot_after.jsonl`、`parameter_update_report.json`、`single_step_update_report.json` | YES | 参数更新链路完整。 |
| checkpoint reload / optimizer audit | 同一 formal 目录 `checkpoint_reload_report.json`、`optimizer_identity_report.json` | YES | 可验证 checkpoint 与 optimizer 状态。 |
| epoch metrics | 同一 formal 目录 `epoch_metrics.csv` | YES | 存在完整训练曲线记录。 |
| 下游冻结 VAL 比较 | `borescope_full_paper_research_archive_20260823/results_snapshot/results/exp12_local_change_downstream/formal_20260823T070600Z/frozen_val_comparison.json` | YES | Route C seed42 VAL Mask mAP50-95 为 0.29352986119614655，相对 Route A -0.005451372933128418。 |
| 下游 training artifacts | 同一 formal 目录 `route_c/summary.json`、`route_c/training_artifacts/results.csv`、`route_c/injection_audit.json`、`route_c/trainer_initialization_audit.json` | YES | 可恢复下游训练与初始化审计。 |
| Exp12 计划/复盘文档 | `borescope_full_paper_research_archive_20260823/experiment_summaries/Exp12/exp12_4_local_change_plan.md`、`exp12_4_5_local_change.md`、`exp12_encoder_analysis.md` | YES | 归档文档齐全。 |

结论：七类 Local Change / Exp12-L 的代码、正式预训练记录、参数审计、proxy IoU 和 downstream seed42 记录仍可恢复。

## 6. 关键结论

1. **气孔 Local Change 成功实现是否仍可恢复？**
   可以。本地 Part3 归档已包含 `08_0_preview_local_change.py`、`08_1b_pretrain_yolo11_local_change_unfrozen.py`、`08_3_train_yolo11_local_change.py`、预训练权重、下游权重、参数审计、checkpoint transfer 审计和下游结果。

2. **七类失败实验是否仍可恢复？**
   可以。Exp09 的归档包含 transfer repair 审计；`transfer_gate_report.json` 明确记录 `REAL_TRANSFER_FAILURE`、`coco_changed_trainable_parameter_count=0`、`witness_pass=false`，可证明早期实验无效。

3. **七类 corrected SSL 实验是否仍可恢复？**
   可以。corrected SimSiam / Exp12-S 的 formal summary 显示 120/120 parameters changed，并保存了 checkpoint/export SHA256、参数快照和下游冻结 VAL 比较。

4. **七类 Local Change / Exp12 是否仍可恢复？**
   可以。Exp12-L 的正式 summary、run config、参数审计、proxy IoU、checkpoint reload 审计和 downstream seed42 冻结 VAL 比较均在归档中。

5. **后续是否真的需要两台服务器同时在线？**
   Phase B 实现差异审计不需要两台服务器同时在线：本地归档加 GitHub 参考提交已足够。
   后续正式实验必须以七类服务器为 `ACTIVE_EXPERIMENT_SERVER`；气孔项目可作为 `REFERENCE_ONLY`。但由于七类服务器当前 SSH alias / 可达性 UNKNOWN，这一点还不是已验证状态。

6. **哪台服务器应该作为主动实验服务器？**
   设计上应使用七类服务器。当前审计只能把 `jetson` 确认为可达但不相关；不能确认它就是七类服务器。不应把不相关 Host 误认为目标服务器。

## 7. 下一阶段建议

READY_FOR_PHASE_B

下一阶段应进行“气孔成功 Local Change 与七类旧 Local Change 的实现差异审计”。重点只做代码/配置/训练机制差异对照，不设计新 generator，不启动实验。
Phase B 可优先使用以下证据源：

- `qikong_ssl_paper_assets/part3_local_change/local_change/config_and_code/`
- `borescope_full_paper_research_archive_20260823/results_snapshot/results/exp12_local_change/formal_20260823T064923Z/`
- `qikong-ssl-det-exp@0b43f124991aec2eeec6ae524592be9dcf81ae12`
- `borescope-new-seg-repro@7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732`

## 8. 安全确认

- 未修改远端文件。
- 未修改 Git 状态。
- 未运行训练、推理或数据预处理。
- 未读取七类 TEST 数据内容。
- 未修改 SSH 配置。
- 未读取私钥、密码、token 或 credential 文件内容。
- 未在报告中输出任何 secret。
