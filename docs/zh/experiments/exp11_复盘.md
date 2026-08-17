# Exp11 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp11：Candidate Freeze 后的一次性 Final TEST

### 1. 为什么做这个实验

所有方法选择必须在 TEST 前结束；最后只估计冻结候选的泛化表现。

### 2. 上一个实验暴露了什么问题

Exp10 排除 Hard Mining 稳健优势，人工选择 seed44 Baseline（最高冻结 Baseline VAL）。

### 3. 本实验核心假设

不再提出改进假设；这是最终测量，不是探索。

### 4. 输入是什么

seed44 Baseline `best.pt`，SHA256 `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`；TEST147 images/285 instances。

### 5. 模型/数据具体怎么处理

Candidate Freeze commit 后，使用 Exp10 同语义 evaluator；标准指标 + fixed conf .25 + size/error/qualitative audit。首次执行在指标前因客户端超时中断，保留证据；用户授权完全同参重试一次。

### 6. 与 baseline 相比唯一改变了什么

只把 split 从冻结 VAL 切到 TEST；模型、seed、checkpoint、参数不变。

### 7. Control variable 是什么

禁止 threshold/model/seed/checkpoint comparison；metrics 生成后 `MODEL_SELECTION_CLOSED=true`。

### 8. 关键参数

imgsz640；fixed conf .25、NMS .70、mask match IoU .50。

### 9. 输出指标

Box/Mask P/R/mAP；TP/FP/FN/F1；size recall；per-class；error taxonomy。

### 10. 实际结果

Box P/R/mAP50/mAP50-95=.662385/.474927/.541636/.293253；Mask=.680727/.498654/.582704/.271621。fixed TP/FP/FN=127/101/158、F1=.495127。VAL .325157→TEST .271621，delta −.053536。

### 11. 如何解释这些结果

存在泛化差距；只能作为观察，不能据此重选 seed 或调参。Crack/Burn/corrosion 是低 AP 类；Tears/Burn/corrosion 是低 Recall 类。

### 12. PASS / FAIL / STOP / Gate

状态代码：PASS / ONE_FINAL_FROZEN_EVALUATION / PROJECT_COMPLETE。Candidate Freeze commit `9991fcfcb9cf6c0ab8920ad7deadeed579ce5585` 在 TEST 前完成；TEST 后无 training/selection。

### 13. 为什么进入下一实验

没有下一实验；`PROJECT_COMPLETE`。

### 14. 这个实验最终在论文/汇报中能说什么

冻结 seed44 Baseline 的一次性 TEST 结果如上。

### 15. 不能说什么

不能因为 TEST 下降重训或改阈值；不能换 seed42/43。

原始证据：`results/final_test/exp11_retry1/`、`docs/handoffs/EXP11_FINAL_PROJECT_REVIEW.md`
