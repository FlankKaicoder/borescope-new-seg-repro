# Exp06 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp06：ROI ResNet18 CE 与 SupCon 表征实验

### 1. 为什么做这个实验

暂时拿掉定位，只给正确/候选 ROI，判断缺陷类别本身是否可分，并寻找过滤低置信 FP 的第二阶段判别器。

### 2. 上一个实验暴露了什么问题

Exp03 说明低阈值能恢复目标但产生 FP；需要独立 ROI 判别能力。

### 3. 本实验核心假设

ROI 分类比全图联合定位分类更容易；SupCon 可改善困难类别表征。

### 4. 输入是什么

同一冻结 patch manifest：train2532、val592，8 类含 background；source-image leakage=0。

### 5. 模型/数据具体怎么处理

ResNet18 ImageNet；CE 50 epochs；SupCon 使用 CE+0.1×监督式对比损失、temperature .07、双视图；同 manifest/batch/sampler/seed。

### 6. 与 baseline 相比唯一改变了什么

CE→SupCon 时仅增加监督式对比表征目标与投影头；数据和分类头任务相同。

### 7. Control variable 是什么

同 patch manifest、batch64、WeightedRandomSampler、seed42、50 epochs；VAL ROI 相同。

### 8. 关键参数

GT bbox 1.2× crop、224×224、ImageNet normalization。

### 9. 输出指标

Accuracy、macro/weighted F1、逐类 P/R/F1、confusion matrix。

### 10. 实际结果

CE accuracy/macro/weighted F1=.635135/.677804/.632591；SupCon=.697635/.693698/.693200，macro F1 Δ=+0.015894。corrosion F1 .3558→.4900；Tip curl support仅3且 F1 .8→.5。

### 11. 如何解释这些结果

多数 ROI 类别具可分性，SupCon 对 ROI representation 有正信号，但类别收益不一致；这不是 segmentation 指标。

### 12. PASS / FAIL / STOP / Gate

状态代码：CE COMPLETE_DIAGNOSTIC；SupCon POSITIVE_ROI_REPRESENTATION / NOT_FINAL_SEGMENTATION_METHOD。SupCon Gate 通过并作为 Exp07 classifier；不再调 SupCon。

### 13. 为什么进入下一实验

将 Exp03 的低阈值候选与 Exp06 classifier 组合成 Exp07 Stage2。

### 14. 这个实验最终在论文/汇报中能说什么

SupCon 改善 ROI 分类 macro F1 0.015894。

### 15. 不能说什么

不能说 SupCon 改善 segmentation；不能用 t-SNE 代替 macro F1。

原始证据：`docs/exp06_*.md`
