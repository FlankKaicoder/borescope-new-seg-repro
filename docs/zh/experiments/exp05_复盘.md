# Exp05 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp05：公平 Hard Mining 单 seed 初步实验

### 1. 为什么做这个实验

让训练更多看到 TRAIN-only hard samples，测试是否改善困难类和 Recall。

### 2. 上一个实验暴露了什么问题

Exp02/03 显示定位、低置信和困难类别问题；Exp04 不支持简单 one-class 解释。

### 3. 本实验核心假设

提高 hard pool 抽样权重可能在相同训练预算下改善 VAL。

### 4. 输入是什么

同一 Exp02 best.pt；TRAIN-only hard pool 201/668。

### 5. 模型/数据具体怎么处理

Control 与 Treatment 都 replacement sampling、每 epoch668、30 epochs；Treatment 仅把 hard weight 从1改为2。

### 6. 与 baseline 相比唯一改变了什么

唯一变量是 hard sample 抽样权重。

### 7. Control variable 是什么

同初始化、同 sampler 形式、同 20,040 sampled images、同331 optimizer steps。

### 8. 关键参数

seed42、30 epochs；normal/hard 1:1 vs 1:2。

### 9. 输出指标

VAL Mask P/R/mAP、fixed errors、困难类指标。

### 10. 实际结果

seed42 Treatment−Control Mask mAP50-95 +0.030354、Recall +0.120856、FP -24、FN 0。

### 11. 如何解释这些结果

这是设计公平的单 seed 初步阳性，不足以证明稳定有效。

### 12. PASS / FAIL / STOP / Gate

状态代码：PRELIMINARY_POSITIVE_SINGLE_SEED；最终被 Exp10 `HARD_MINING_NOT_CONFIRMED` 覆盖。当时 `POSITIVE_CANDIDATE`；最终结论必须服从 Exp10 三 seed。

### 13. 为什么进入下一实验

一方面进入 ROI/Stage2 路线，另一方面最终由 Exp10 验证跨 seed 稳健性。

### 14. 这个实验最终在论文/汇报中能说什么

seed42 出现 +0.030354 初步增益。

### 15. 不能说什么

不能说 Hard Mining 稳定提升 3 个点。

原始证据：`docs/exp05_hard_mining_fast_repro.md`
