# 最近里程碑交接

## 协作体系整改交付（2026-11-23，Asia/Shanghai）

用户批准前序完整方案并要求实施。本次授权限于协作文档整改、验证、Git commit 和正常 push；不包含模型训练、数据或标签修改、推理评估、付费调用、论文仓库操作或追加科研实验。

| 项目 | 交接事实 |
| --- | --- |
| 适用分支 | `codex/research-collaboration-workflow`，基于 `phase/exp14-pseudonormal-region` |
| 整改基线／已有结果提交 | `83274732f72133cbe9a3abff2ab9a6c66c0e2eea` |
| 正式训练及评估代码提交 | `672e672d528a58570f2d39b2372f727f422f163b` |
| 正式训练结果提交 | `a01a3a3f43edae891298f5e02b6cdd11e9738811` |
| 本次交付提交 | 提交后用 `git log -1 --format=%H -- docs/latest_handoff.md` 解析；最终回复提供提交号与 push 回执 |
| 当前权威科研状态 | [PROJECT_STATE](PROJECT_STATE.md#current-authoritative-snapshot) |
| 治理依据 | [GOV-20261123-01](DECISION_LOG.md#gov-20261123-01) |

### 已完成内容与验证

- 用 [AGENTS](../AGENTS.md) 明确三方职责、Codex 工程自主权、科研暂停条件、Plan／Goal／Code 工作流程及收尾责任。
- 复用八个已有文件；仅新增本交接入口。README 负责导航，PROJECT_STATE 负责当前事实，ROADMAP 负责未来计划，DECISION_LOG 负责重大决定，experiment_index 负责证据索引。
- 当前摘要与历史快照分开；旧实验、原始记录、结果和两份 registry 保留。历史决策 D043／D044 通过范围消歧，不重排旧编号。
- 工程验收 PASS：改动严格限于 8 个既有文档和本新增入口；`git diff --check` 通过，51 个新增本地链接及锚点可解析，文件为 UTF-8 无 BOM。
- 证据保全 PASS：PROJECT_STATE、DECISION_LOG、experiment_index 的原历史正文完整保留；HISTORICAL_METHODS 仅修正一句，原 README 最终指标表和权重事实保留；模型代码、配置、数据、结果、两份 registry、旧报告及 handoff 无差异。
- 定向对账 PASS：20／20 个冻结代码文件按原执行身份的 CRLF→LF 规则计算 SHA256 一致；最新 VAL 指标及 FP／FN 与实际 CSV 一致。未运行训练或评估。
- 工程交叉审阅 PASS：自主修复、协议变更暂停、负向结果记录、TEST 使用、Compact 恢复和 push 失败处理规则明确。本项属于工程自审，不替代待进行的 ChatGPT 科研结果审计。

本分支的规范变更未自动合并到 main 或其他实验分支。main 的四个本地未推送提交和其他工作树均未处理；后续推广须在目标分支核对实际状态，不能复制本分支的科研摘要冒充全局现状。

## 最近科研里程碑：Exp14 seed42 冻结 VAL

稳定报告：[E0／E1-FULL seed42 VAL 交接](exp14/EXP14_E0_E1_FULL_SEED42_VAL_HANDOFF.md)。原始报告保持不变，不将本次工程自审视为 ChatGPT 独立科研审计。

| 状态 | 内容 |
| --- | --- |
| 实验执行 | seed42 两臂各 100 轮，冻结 VAL 已完成 |
| 独立科研审计 | `EXP14_E0_E1_FULL_SEED42_VAL_COMPLETE_PENDING_CHATGPT_RESULT_GATE`，尚待结果审计 |
| 主指标 | Mask mAP50–95：E0 `0.24951245667305366`，E1-FULL `0.23039514532654876`，差 `−1.9117311346504906` 个百分点 |
| 固定置信度计数 | Mask FP `57 → 66`，FN `201 → 219` |
| 预算限制 | 等 100 轮、原图各 66,800 张次；有效更新 `2,100／3,300`，E1 另有修复图 37,900 张次 |
| 数据边界 | A10 冻结 379 张 A 图，13 条同资产标签冲突隔离；B 区域候选不等于可信 Pair，尚未准入 |
| TEST 范围 | Exp14 该次运行未使用 TEST；Exp11 历史正式 TEST 已完成，不得表述为项目从未访问 TEST |

证据：[VAL 汇总](../results/exp14_e0_e1_full_seed42_val/summary.json)、[整体指标](../results/exp14_e0_e1_full_seed42_val/overall_metrics.csv)、[固定点计数](../results/exp14_e0_e1_full_seed42_val/fixed_counts.csv)、[训练汇总](../results/exp14_e0_e1_full_seed42_train/summary.json)、[冻结协议](exp14/EXP14_E0_E1_FULL_FORMAL_PROTOCOL.md)、[A10 数据交接](exp14/EXP14_A10_A_DATASET_FREEZE_HANDOFF.md)。

本地可核验汇总、表格及冻结代码身份；完整逐批日志、权重和授权回执位于私有目录，本次未重新核验。最近运行报告记录单张 RTX 3080 Ti；本次没有远端硬件探测。单种子、等轮数且不等更新数的负向结果，不构成对全部伪正常方法的最终否定。

## 下一步与恢复

- 工程：核对本分支的交付提交及远端同步状态；无需为本次治理整改新增实验或运行历史发布脚本。
- 科研：向 ChatGPT 提供固定结果提交 `83274732f72133cbe9a3abff2ab9a6c66c0e2eea` 的上述报告和证据，独立审计后由用户决定研究行动。
- 更新预算匹配 E0、配对多种子和 B 路线仍是候选；本次批准不解除既有冻结协议、执行器授权或付费请求锁。
- 本任务未启动训练、评估或付费 job，无运行进程、已消耗调用额度或恢复中的实验。未完成工作应先 Compact 并核对实际任务状态，避免重复执行。
