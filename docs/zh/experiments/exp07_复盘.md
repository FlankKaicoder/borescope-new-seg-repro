# Exp07 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp07：YOLO + ROI classifier Stage2 系统

### 1. 为什么做这个实验

把 Exp03 的高 Recall 候选与 Exp06 的 ROI 判别器结合，尝试过滤 FP 或纠正类别。

### 2. 上一个实验暴露了什么问题

直接降 Stage1 threshold 会 FP explosion，ROI classifier 又显示一定可分性。

### 3. 本实验核心假设

Stage2 可在保留低置信 TP 的同时删除 FP；Mode B 还可重分类。

### 4. 输入是什么

冻结 Exp02 Stage1、冻结 SupCon classifier、冻结 VAL。

### 5. 模型/数据具体怎么处理

Stage1 conf .05/.10/.15；Stage2 threshold .3/.5/.7。Mode A 按 defect probability 过滤；Mode B 按预测类概率过滤并重分类；mask 始终来自 Stage1。

### 6. 与 baseline 相比唯一改变了什么

只加冻结 classifier 的过滤/重分类规则；不生成新 mask。

### 7. Control variable 是什么

网格预定、无阈值扩展；与 YOLO conf .25 fixed point 比；AP evaluator 不可靠所以 AP=N/A。

### 8. 关键参数

最佳 A=.15/.3；最佳 B=.15/.7。

### 9. 输出指标

TP/FP/FN、P/R/F1、wrong-class、low-conf retention、latency。

### 10. 实际结果

baseline F1=.474903；Mode A=.452336、FP118/FN175；Mode B=.450820、FP82/FN186。91 个 recoverable 中 70 个在 Stage1 .15 前已不可用；A/B 分别保留16/15。

### 11. 如何解释这些结果

Mode B 确实减少 FP，但增加 FN、降低 Recall；Stage2 规则未形成净 F1 增益。

### 12. PASS / FAIL / STOP / Gate

状态代码：NEGATIVE。`NEGATIVE`；HardMining+FrozenStage2 probe 未过 Gate。

### 13. 为什么进入下一实验

尝试把 classifier knowledge 回灌单阶段网络的想法导向 Exp08 KD。

### 14. 这个实验最终在论文/汇报中能说什么

当前两阶段 reconstruction 在固定规则下未超过 baseline。

### 15. 不能说什么

不能虚构 Stage2 mAP；不能说 ROI classifier 本身无效。

原始证据：`docs/exp07_stage2_fast_repro.md`
