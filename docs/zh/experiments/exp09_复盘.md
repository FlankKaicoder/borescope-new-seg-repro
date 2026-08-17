# Exp09 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp09：SimSiam domain adaptation 与参数/transfer 有效性审计

### 1. 为什么做这个实验

探索 COCO 预训练与孔探域之间的 domain gap，尝试在 TRAIN 图像上自监督适配 YOLO backbone。

### 2. 上一个实验暴露了什么问题

前序方法显示表征和定位仍困难，且不希望访问标签外的 TEST。

### 3. 本实验核心假设

SimSiam 的双视图一致性可让 backbone 学到孔探域表征。

### 4. 输入是什么

668 TRAIN images only；官方 COCO backbone。

### 5. 模型/数据具体怎么处理

100 epoch SimSiam；检查 loss、feature/embedding std；随后做参数 delta、key/shape、in-memory load、FP32/native round-trip 与 BN 行为审计。

### 6. 与 baseline 相比唯一改变了什么

加入自监督训练目标；不使用 VAL/TEST 做 SSL 选择。

### 7. Control variable 是什么

TRAIN-only；downstream Gate 必须先证明 trainable backbone 真正改变。

### 8. 关键参数

100 epochs、batch32；transfer expected tensors240、trainable params120。

### 9. 输出指标

loss/std finite、changed trainable params、BN buffers、transfer coverage。

### 10. 实际结果

曲线 finite 且无明显 collapse，但相对 COCO：0/120 trainable parameters changed，120/120 BN buffers changed。Exp09.2a 证明 transfer 240/240、immediate trainable120/120、round-trip240/240，`PASS_REVISED`。

### 11. 如何解释这些结果

训练表面正常不证明发生了参数学习；问题在 SSL implementation/optimization，而非 transfer。没有有效 adapted backbone，就不能合法评价 downstream。

### 12. PASS / FAIL / STOP / Gate

状态代码：INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED。reconstruction=`INVALID_BY_BACKBONE_NO_UPDATE`；downstream=`NOT_RUN_BY_GATE`；performance=`NOT_EVALUATED`。

### 13. 为什么进入下一实验

停止 SSL 路线，回到已有最有希望的 Hard Mining，进入 Exp10 多 seed 核验。

### 14. 这个实验最终在论文/汇报中能说什么

本次 SimSiam reconstruction 无有效 trainable backbone update。

### 15. 不能说什么

不能说 SimSiam 方法无效、降低 mAP 或 downstream 失败。

原始证据：`docs/exp09_*.md`
