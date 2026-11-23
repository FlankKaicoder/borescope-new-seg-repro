# Roadmap

研究计划以本文件登记，当前执行事实与授权以 [PROJECT_STATE](docs/PROJECT_STATE.md) 为准。候选计划不构成实验授权；本次核验仅覆盖当前分支，不能据此判断其他分支实验是否完成或是否已有结果。

## 已完成：Exp00–Exp11 复现与最终评估

- [x] Exp00 environment, schema/pair, class/polygon/scale, duplicate/leakage audit
- [x] Exp01 JSON → YOLO-seg, group-aware split, Dataset v1 freeze
- [x] Exp02 baseline, numerical diagnosis and size/error audit
- [x] Exp03–Exp09 historical-method reconstruction and engineering/validity Gates
- [x] Exp10 controlled three-seed verification; Hard Mining not confirmed
- [x] Full Dataset Review and RQ1–RQ10 evidence consolidation
- [x] Option A direct finalization selected; no more model experiments within Exp00–Exp11
- [x] Candidate Freeze: Baseline / seed44 / TEST not accessed at freeze
- [x] Exp11.0 final unified TEST
- [x] Exp11.1 qualitative/error audit
- [x] Exp11.2 paper/thesis/project materials
- [x] Exp00–Exp11 experimental phase complete (`PROJECT_COMPLETE`)

## 已完成：Exp12 独立表示学习研究

- [x] Exp12.0 independent SimSiam encoder/audit design (`NO_TRAINING`)
- [x] Exp12.0 runtime structure and TRAIN-only probe (`PASS_NO_TRAINING`)
- [x] Exp12.1-S single-step and 1 epoch smoke (`PASS`; backbone update 120/120)
- [x] Exp12.1 formal 100-epoch TRAIN-only SimSiam (`PASS`; no collapse)
- [x] Exp12.2 final parameter/checkpoint/export audit (`PASS`; 120/120 changed)
- [x] Exp12.3-S downstream fairness/Trainer-initialization smoke (PASS)
- [x] Exp12.3 Route A/B 100-epoch frozen-VAL comparison (PASS; single-seed positive signal)
- [x] Exp12.4 multi-scale local-change SSL smoke and 30-epoch formal audit (PASS; backbone update 120/120)
- [x] Exp12.5 Route C 100-epoch frozen-VAL comparison (PASS; no Mask mAP50-95 gain; TEST untouched)

## 后续已核验里程碑

- [x] Exp13 历史生成、修复与几何方法研究已归档；云端 G3-D2 以部分人工证据关闭，未建立正式 TRE Gate。入口见 [实验索引](docs/experiment_index.md) 和 [G3-D2 交接](docs/exp13_cloud/G3_D2_HANDOFF.md)。
- [x] Exp14 A 数据整理、冻结与训练 smoke 完成；[E0／E1-FULL 正式协议](docs/exp14/EXP14_E0_E1_FULL_FORMAL_PROTOCOL.md) 和 [seed42 训练交接](docs/exp14/EXP14_E0_E1_FULL_SEED42_TRAIN_HANDOFF.md) 保留执行证据。
- [x] Exp14 seed42 冻结 VAL 完成：E0／E1-FULL Mask mAP50–95 为 0.249512／0.230395，差 −1.912 个百分点；仅为当前等 epoch、单种子比较，见 [结果交接](docs/exp14/EXP14_E0_E1_FULL_SEED42_VAL_HANDOFF.md)。

## 当前工作与未来候选

状态使用“候选未批准／已批准／进行中／完成”；“已批准”需能定位到用户决定及其范围，执行进度另记于当前状态。

| 工作或研究项 | 状态 | 边界与目的 |
|---|---|---|
| 科研协作文档整改 | 完成（文档验收通过） | 仅协作规则、项目记忆和交接入口；不运行实验、不改科研证据；交付提交及同步状态见最近交接与最终汇报 |
| E0-UPDATE 更新预算匹配控制 | 候选未批准 | 检验 2,100／3,300 次有效更新差异的影响；原协议预留该对照，本轮未授权执行 |
| E0／E1-FULL 配对多种子重复 | 候选未批准 | 检验当前单种子观察的稳定性，需明确实验方案、预算和结果口径 |
| Exp14 B 路线修复配对研究 | 候选未批准 | 需独立方法方案、配对准入与用户批准；A 路线完成不自动授权 B 路线 |

当前未因本次协作整改新增任何模型实验授权。后续科研方向由用户决定，ChatGPT 独立设计与审计，Codex 在批准范围内自主实现和验证。
