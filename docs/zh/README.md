# 新孔探项目中文学习与汇报材料

本目录是 Exp00–Exp11 完成后的中文学习层，不替代原始实验记录。所有事实以 docs/PROJECT_STATE.md、Exp10 最终 Review、Exp11 final results 和 CHANGELOG.md 为准。

## 推荐阅读顺序

1. [01 项目全流程串讲](01_项目全流程串讲.md)：先理解“为什么下一步自然发生”。
2. [02 实验逐项详细复盘](02_实验逐项详细复盘.md)：按统一模板学习控制变量、Gate 与结果边界。
3. [03 实验结果与研究结论](03_实验结果与研究结论.md)：区分 FINAL、MULTI-SEED、DIAGNOSTIC、GATE、INVALID。
4. [05 图表与可视化索引](05_图表与可视化索引.md)：准备图表讲解。
5. [04 汇报与答辩口径](04_汇报与答辩口径.md)：练习 1/3/15 分钟版本和追问。
6. 遇到细节再回到 [experiments](experiments/) companion 与原始 docs/expXX_*.md。

## 绝对边界

- 项目状态：PROJECT_COMPLETE。
- 最终方法：YOLO11n-seg Baseline，seed44。
- Hard Mining：HARD_MINING_NOT_CONFIRMED。
- KD：SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED。
- SimSiam：INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED，不是“SimSiam 方法无效”。
- 本轮只整理文档与已有证据可视化：无训练、无推理、无 TEST 重新访问、无模型/阈值选择。
