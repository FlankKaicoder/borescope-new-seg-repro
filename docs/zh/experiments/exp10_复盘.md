# Exp10 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp10：三随机种子预算匹配最终验证

### 1. 为什么做这个实验

Exp05 只有 seed42；单 seed 的 +0.030354 可能是随机性。

### 2. 上一个实验暴露了什么问题

Hard Mining 是唯一初步阳性的 segmentation candidate，但证据等级不足。

### 3. 本实验核心假设

若方法稳定，Treatment−Control 应在 seeds42/43/44 大多为正且平均为正。

### 4. 输入是什么

每个 seed 的 Baseline、Uniform Control、Hard Treatment；冻结 TRAIN hard pool 与统一 VAL evaluator。

### 5. 模型/数据具体怎么处理

三 seed；Baseline100；Control/Treatment 均从同 seed baseline 继续30 epoch、20,040 samples、331 steps；统一重跑九个 checkpoint VAL。

### 6. 与 baseline 相比唯一改变了什么

Treatment 相对 Control 唯一变量仍是 hard weight；seed 是重复实验维度。

### 7. Control variable 是什么

预算、初始化、sampler、evaluator 语义匹配；历史 seed43 nonfinite run 保留，受控 restart 独立记录。

### 8. 关键参数

seeds42/43/44；hard weight1→2；AMP true；imgsz640；batch32。

### 9. 输出指标

paired Mask mAP50-95/Recall/F1/FP/FN delta；均值和 sample std。

### 10. 实际结果

paired mAP50-95 Δ：+0.030354、−0.006673、−0.040311；1/3 positive；mean −0.005543±0.035346。Baseline mean .307881±.014963。

### 11. 如何解释这些结果

Exp05 的单 seed 正信号未被复现；随机种子敏感性足以改变结论。Uniform Control 不能 post-hoc 晋升候选。

### 12. PASS / FAIL / STOP / Gate

状态代码：COMPLETE / HARD_MINING_NOT_CONFIRMED。`HARD_MINING_NOT_CONFIRMED`；推荐 Baseline 进入人工 Candidate Freeze。

### 13. 为什么进入下一实验

在任何 TEST 前冻结最终方法、seed、checkpoint 与选择规则。

### 14. 这个实验最终在论文/汇报中能说什么

多 seed 预算匹配结果不支持 Hard Mining 稳健增益。

### 15. 不能说什么

不能继续引用 seed42 “提升3点”作为最终结论。

原始证据：`docs/exp10_*.md`
