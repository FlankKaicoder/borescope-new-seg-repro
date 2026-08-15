# 新数据集实验时间线（Exp00–Exp10）

本时间线只保留 Purpose、Result、Gate 和 What it changed。全部模型结果来自既有记录；`test_accessed=false`。

| 阶段 | Purpose | Result | Gate | What it changed |
|---|---|---|---|---|
| Exp00 | 审计环境、schema、配对、类别、polygon、重复与来源线索 | 969 监督图/1847 实例/7 类；24 unpaired；88 near groups；0 polygon issue | 数据 blocker 经权威政策关闭 | 建立专业 GT 不改写、unpaired 排除、near-group 防泄漏原则 |
| Exp01 | 转换并冻结 leakage-safe 数据集 | 668/154/147；0 cross-split group leakage；hash 冻结 | Dataset Freeze PASS | 后续全部实验使用同一 v1/split |
| Exp02.0 | 确定可行训练配置 | batch32 smoke/probe PASS | Smoke Gate PASS | 冻结 640/batch32 基础配置 |
| Exp02.1/2a | 建立 YOLO11n-seg baseline 并解释 early NaN | 100 epochs；seed42 Mask mAP50-95 0.298981；C2PSA AMP overflow 定位 | PASS_WITH_NUMERICAL_WAIVER | 获得有效 reference，同时保留数值限制 |
| Exp02.2/2.3 | 审计 size/error 并决定是否升分辨率 | Recall 不随 size 单调；定位错误重要 | Exp02.3 DEFERRED_BY_EVIDENCE | RQ2 保留为 optional unanswered question |
| Exp03 | 验证低置信度可恢复现象 | 91/173 FN recoverable；低阈值 FP explosion | POSITIVE_DIAGNOSTIC | 支持候选感知存在，不支持直接降阈值 |
| Exp04 | 检验 Crack 是否主要受多类竞争影响 | one-class 无清晰增益 | NO_CLEAR_GAIN | 更支持定位/表征/内在难度解释 |
| Exp05 | 单种子公平 Hard Mining | seed42 paired mAP +0.030354 | POSITIVE_CANDIDATE | 形成待多种子验证候选，不构成最终结论 |
| Exp06.1 | 给定 ROI 后评估分类可行性 | CE macro F1 0.677804 | COMPLETE_DIAGNOSTIC | 部分支持分类比联合定位更容易 |
| Exp06.2 | 检验 SupCon ROI 表征 | macro F1 0.693698；delta +0.015894 | SUPCON_POSITIVE | 证明 ROI representation gain，不外推 segmentation |
| Exp07 | 将 ROI classifier 接入 Stage2 | Mode A/B F1 均低于 YOLO fixed point | NEGATIVE | 关闭当前 Stage2 decision-rule 路线 |
| Exp08 | reconstruction classifier-teacher KD | batch32 single step >70 s；未正式训练 | SKIPPED_BY_ENGINEERING_GATE | 记录工程可行性边界，不评价 KD 效果 |
| Exp09/9.2a | reconstruction SimSiam domain adaptation 与 transfer | SSL finite/no collapse；transfer PASS；trainable params changed 0/120 | INVALID_BY_BACKBONE_NO_UPDATE | downstream 被阻止；性能 NOT_EVALUATED |
| Exp10 | 三种子、预算匹配验证 Hard Mining | paired delta +0.030354/-0.006673/-0.040311；mean -0.005543±0.035346 | HARD_MINING_NOT_CONFIRMED | Exp10 覆盖 Exp05 单种子候选；等待后续人工决策 |
| Review10.5 | 全项目证据合并和后续决策准备 | 状态/结果/ROI/数据/未来候选矩阵完成 | NO TRAINING / NO TEST | 项目进入 WAITING_NEXT_EXPERIMENT_DECISION |
