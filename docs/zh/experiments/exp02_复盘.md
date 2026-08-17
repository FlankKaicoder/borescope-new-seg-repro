# Exp02 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp02：YOLO11n-seg Baseline、尺度/错误审计与 AMP 数值根因

### 1. 为什么做这个实验

先建立统一单阶段 segmentation 参照，后续任何方法都必须回答“相对 baseline 改了什么”。

### 2. 上一个实验暴露了什么问题

Exp01 只证明数据可加载，尚不知道模型表现、错误类型和硬件可承受 batch。

### 3. 本实验核心假设

轻量 YOLO11n-seg 在 640 输入和单张约 22GB 2080 Ti 上能稳定完成 100 epoch，并提供足够的错误信号。

### 4. 输入是什么

Dataset v1；官方 COCO `yolo11n-seg.pt`。

### 5. 模型/数据具体怎么处理

先做 batch 8/16/24/32 满批 probe 与完整一轮 smoke；冻结 batch32 后训练 seed42 100 epoch；独立 VAL；按 TRAIN-only mask-area 分位数审计尺度；定位 early VAL NaN。

### 6. 与 baseline 相比唯一改变了什么

首次引入监督式实例分割模型；其余数据与 split 均冻结。

### 7. Control variable 是什么

imgsz640、AdamW、AMP、deterministic、seed42；TEST untouched；size thresholds 只由 TRAIN 生成。

### 8. 关键参数

epochs100、batch32、imgsz640、conf .001 标准 VAL；fixed audit conf .25、NMS .70、mask IoU .50。

### 9. 输出指标

Box/Mask P、R、mAP50、mAP50-95；TP/FP/FN；size recall；loss finite audit。

### 10. 实际结果

seed42 VAL Mask mAP50-95=0.298981；fixed TP/FP/FN=123/99/173、F1=0.474903。epoch1–5 VAL losses NaN，epoch6 后恢复；FP16 C2PSA qk matmul overflow 被定位，FP32 同批 finite；checkpoint tensors finite。

### 11. 如何解释这些结果

最终模型和标准指标有效，但违反原始“全程无 NaN”硬条件；因此必须显式 waiver，不能悄悄忽略。尺度 recall 非单调，large 反而最低。

### 12. PASS / FAIL / STOP / Gate

状态代码：PASS_WITH_NUMERICAL_WAIVER。用户接受 `PASS_WITH_NUMERICAL_WAIVER`；未重训美化历史。

### 13. 为什么进入下一实验

173 个 FN 中许多疑似低置信，导向 Exp03；Crack/Burn/corrosion 困难导向 Exp04/05。

### 14. 这个实验最终在论文/汇报中能说什么

建立了 baseline，并完成 AMP 数值根因与错误/尺度审计。

### 15. 不能说什么

不能说训练全程数值完全正常；不能说小目标必然最差。

原始证据：`docs/exp02_*.md`
