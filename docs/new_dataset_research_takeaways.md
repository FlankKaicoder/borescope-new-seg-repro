# 新数据集研究结论：论文、面试与项目陈述版本

## A. Dataset engineering

- 对 993 图/969 JSON 做 schema、配对、polygon、hash 和视觉近重复审计，最终冻结 969 张专业监督样本与 1847 个实例。
- 24 张无 JSON 图片明确排除，不能默认当 background；专业 GT 不以自动多数表决改写。
- 将 88 个近重复连通组作为不可拆 split 单元，得到 668/154/147 group-aware split 和 0 cross-split leakage。
- 用 split、class mapping、raw manifest SHA256 固定数据 provenance；这比随机图片级划分更能防止相邻视角泄漏。

## B. Segmentation baseline

- YOLO11n-seg 640/100 epochs 建立可复核 baseline；三种子 Mask mAP50-95 为 `0.307881 ± 0.014963`。
- early validation NaN 没有被忽略：根因定位到 C2PSA FP16 qk overflow；train loss、最终权重和 independent VAL 有效，状态准确保留为 `PASS_WITH_NUMERICAL_WAIVER`。
- 不把 VAL 写成 TEST；Candidate Freeze 和 TEST 都尚未执行。

## C. Error analysis

- 173 FN 中 91 个（52.6%）有低置信度可恢复候选，但简单降低阈值会造成 FP explosion。
- baseline error 不只来自类别不均衡：定位、无响应、mask quality 和分类混淆共同存在。
- 尺度 Recall 不呈单调规律，因此“所有小目标都应靠更高分辨率解决”尚未被支持。

## D. Representation learning

- 给定 GT ROI 后，ResNet18 CE macro F1=0.677804，说明多数类别具备可学习的局部表征；corrosion/background 仍困难。
- SupCon 将 ROI macro F1 提升到 0.693698，并改善 Burn/Crack/corrosion/background Recall。
- 正确表述是 “SupCon improves ROI classification representation”，不是 “SupCon improves segmentation”。

## E. System design

- Stage2 classifier 可以过滤 FP，但当前 decision rule 无法同时保持 Recall，最终 fixed-point F1 未超过 YOLO。
- 这说明模块级表征提升不等于端到端系统提升；候选生成、score calibration 和决策规则同样关键。
- Uniform Control 表明 continued training 本身可能影响结果，所以 Hard Mining 的方法效果必须对 budget-matched Control 计算。

## F. Experimental methodology

- 单种子 seed42 Hard Mining 提示 +0.030354，但三种子配对均值为 `-0.005543 ± 0.035346`，正增益仅 1/3；最终结论必须是 `HARD_MINING_NOT_CONFIRMED`。
- 公平比较固定初始化、采样量与 optimizer steps；多随机种子验证避免把偶然结果包装成稳定增益。
- 数值 Gate、checkpoint SHA、有限性检查、artifact decode 和 Git evidence 共同组成 reproducibility chain。

## G. Failed-method lessons

- Negative 不等于无价值：One-class 缩小了 Crack 机制假设；Stage2 揭示 FP/Recall trade-off；Hard Mining 展示 seed sensitivity。
- Exp08 是 engineering-gated，不是 KD negative；Exp09 是 invalid reconstruction，不是 SimSiam negative。
- 及时停止高成本或无效路线，是研究工程的一部分；不能通过补造数字掩盖 Gate。

## 可直接使用的总结句

本项目的主要贡献不是某个单点“涨分”，而是一条从数据治理、泄漏控制、baseline 数值诊断、错误机制、表征学习、系统负结果到三随机种子验证的完整可复现研究链。最终最强结论是：当前 Hard Mining 收益不稳健；ROI SupCon 的表征收益成立但没有自动转化为 segmentation/system gain；在现有毕业设计范围内，直接完成论文比继续盲目调参更有价值。
