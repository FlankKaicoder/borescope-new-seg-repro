# Exp03 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp03：Low-confidence recovery 诊断

### 1. 为什么做这个实验

判断 baseline 的 FN 是“完全没响应”，还是已有低分候选被 operating threshold 截掉。

### 2. 上一个实验暴露了什么问题

Exp02 在 conf .25 有 173 FN，Recall 0.4155。

### 3. 本实验核心假设

降低阈值会找回一部分真实目标，但也可能造成大量 FP。

### 4. 输入是什么

冻结 Exp02 seed42 checkpoint 与冻结 VAL。

### 5. 模型/数据具体怎么处理

仅在预定 12 个阈值上复用 VAL prediction 诊断；对 173 FN 分成 LOW_CONF_RECOVERABLE、WRONG_CLASS、LOCALIZATION_FAILURE、NO_RESPONSE。

### 6. 与 baseline 相比唯一改变了什么

只改变评估 operating threshold，不训练模型。

### 7. Control variable 是什么

同 checkpoint、同 VAL、同 matching 规则；不访问 TEST。

### 8. 关键参数

F1 最佳仍为 conf=.25；recall95 点为 .005。

### 9. 输出指标

P/R/F1、TP/FP/FN、FN taxonomy。

### 10. 实际结果

91/173 FN（52.6%）为 LOW_CONF_RECOVERABLE；但 conf=.005 时 FP=4569、F1=0.08049。

### 11. 如何解释这些结果

低分响应真实存在，但“直接降阈值”不可作为最终方案；需要更可靠的候选过滤/判别。

### 12. PASS / FAIL / STOP / Gate

状态代码：POSITIVE_DIAGNOSTIC / NOT_FINAL_MODEL。`POSITIVE_DIAGNOSTIC`，不晋升为模型。

### 13. 为什么进入下一实验

困难 Crack 是否源于多类竞争？进入 Exp04；FP 过滤思路也为 Exp06/07 铺路。

### 14. 这个实验最终在论文/汇报中能说什么

确认了低置信恢复现象及其 FP 代价。

### 15. 不能说什么

不能说降低阈值提高了最终模型。

原始证据：`docs/exp03_low_conf_fast_repro.md`
