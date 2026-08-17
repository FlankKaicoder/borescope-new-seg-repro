# Exp08 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp08：Classifier-teacher→YOLO KD 工程探针

### 1. 为什么做这个实验

若 Stage2 推理链成本高且会误删 TP，可尝试把 classifier knowledge 作为训练辅助信号回灌 YOLO。

### 2. 上一个实验暴露了什么问题

Exp07 组合系统 negative，但 Exp06 classifier 有判别能力。

### 3. 本实验核心假设

冻结 teacher 的 ROI logits 可监督 YOLO 中间特征辅助头。

### 4. 输入是什么

冻结 SupCon teacher；官方 yolo11n-seg student；TRAIN smoke。

### 5. 模型/数据具体怎么处理

YOLO layer4/P3 stride8、128通道；GT boxes→ROIAlign7×7→七类 auxiliary head；teacher 1.2× ROI 224。

### 6. 与 baseline 相比唯一改变了什么

新增辅助 CE/KD 路径；尚未进入 formal training。

### 7. Control variable 是什么

teacher eval/no grad；先 smoke，Gate 未过不得正式训练。

### 8. 关键参数

batch32、teacher crop chunk16；候选 lambda/T 未搜索。

### 9. 输出指标

显存、单步时间、forward/loss finite。

### 10. 实际结果

batch4 在线 teacher ROI 曾 OOM；chunking 后 batch32 finite，但单 training step >70s。

### 11. 如何解释这些结果

当前实现工程成本不可接受；这没有回答 KD 方法效果。

### 12. PASS / FAIL / STOP / Gate

状态代码：SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED。`SKIPPED_BY_ENGINEERING_GATE`；AUX_CE/KD formal runs 和 VAL comparison 均未执行。

### 13. 为什么进入下一实验

不修 KD；转向独立的 domain adaptation 扩展 Exp09。

### 14. 这个实验最终在论文/汇报中能说什么

KD reconstruction 被工程 Gate 停止、NOT_EVALUATED。

### 15. 不能说什么

不能说 KD failed 或降低精度。

原始证据：`docs/exp08_kd_fast_repro.md`
