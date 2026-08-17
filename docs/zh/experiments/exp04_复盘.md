# Exp04 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp04：Crack one-class 诊断

### 1. 为什么做这个实验

把定位/表征困难与多类别竞争区分开：若只保留 Crack 后明显变好，说明 class competition 可能是主因。

### 2. 上一个实验暴露了什么问题

Crack 在 baseline 的 Recall/AP50-95 很低，并兼有 confidence 与 classification 错误。

### 3. 本实验核心假设

去掉其他缺陷类别竞争可能提升 Crack。

### 4. 输入是什么

保持 train668/val154 membership，仅 Crack polygon 作为 positive，其余图像作为 background。

### 5. 模型/数据具体怎么处理

YOLO11n-seg one-class 100 epoch；与 multi-class baseline 的 Crack 指标对照。

### 6. 与 baseline 相比唯一改变了什么

唯一主要改变为监督类别空间；图像 membership 不变。

### 7. Control variable 是什么

同 split membership、同训练长度；不创建 test。

### 8. 关键参数

100 epochs；Crack-only train 95 instances、val24。

### 9. 输出指标

Crack Mask Recall/AP50/AP50-95。

### 10. 实际结果

one-class .16667/.18352/.05548；multi-class .27883/.17325/.04675；Recall -0.11216，AP50-95 仅 +0.00873。

### 11. 如何解释这些结果

没有清晰增益，不支持“多类竞争是 Crack 主瓶颈”；定位、表征或固有难度更值得关注。

### 12. PASS / FAIL / STOP / Gate

状态代码：NO_CLEAR_GAIN。`NO_CLEAR_GAIN`，停止 one-class 路线。

### 13. 为什么进入下一实验

转向从已知错误中构造 hard pool，进入 Exp05。

### 14. 这个实验最终在论文/汇报中能说什么

one-class 诊断未发现清晰优势。

### 15. 不能说什么

不能说 Crack one-class 确定变差或多类竞争完全无影响。

原始证据：`docs/exp04_crack_oneclass_fast_repro.md`
