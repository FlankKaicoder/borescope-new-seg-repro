# Decision log

| Date | Decision ID | Question | Decision | Reason | Affected experiments |
|---|---|---|---|---|---|
| 2026-08-12 | D001 | 是否复现旧 9 类实验？ | 不复现。 | 当前研究主线是新数据集的实例分割与历史四类任务的方法思想重建。 | All |
| 2026-08-12 | D002 | 是否复现 GAN、CutPaste、DRAEM、医学息肉迁移？ | 不复现。 | 属于明确排除的外围路线。 | All |
| 2026-08-12 | D003 | 当前主任务是什么？ | 以 instance segmentation 为主。 | 新数据提供 polygon 标注，研究问题围绕定位、mask 与分类。 | Exp01+ |
| 2026-08-12 | D004 | 旧 537 张四类任务如何使用？ | 只作为历史方法思想来源，不追求旧数字或 bit-exact reproduction。 | Exp02–Exp08 |
| 2026-08-12 | D005 | SimSiam 的历史定位是什么？ | 后续新增扩展，不冒充旧 537 张实验。 | Exp09 |
| 2026-08-12 | D006 | GPU 规划采用哪种事实？ | 当前单张约 22GB RTX 2080 Ti 的实测拓扑覆盖初始“双 2080 Ti × 11GB”假设。 | Environment and all training |
| 2026-08-12 | D007 | 当前能否进入 Exp01？ | 不能；Data Gate STOP。 | Exp00.4/00.5, Exp01 |
| 2026-08-12 | D008 | 专业 JSON 是否需要继续人工复核或自动统一类别？ | 不需要。全部专业 JSON 作为权威 Ground Truth，原标签保持不变。 | Exp00.5, Exp01+ |
| 2026-08-12 | D009 | 24 张无 JSON 图片如何处理？ | 统一记录为 `excluded_unpaired`，不删除、不补空 JSON、不当作 background、不进入任何 split。 | Exp00.4, Exp01 |
| 2026-08-12 | D010 | 22 个 near-duplicate cross-label groups 如何处理？ | 保留全部原专业标签；连通组不可拆分，仅作为 split 防泄漏约束和后续混淆分析证据。 | Exp01+ |
| 2026-08-12 | D011 | 本轮允许推进到哪里？ | Data Gate PASS FOR EXP01；只执行 Exp01.0--01.2 和 Dataset Freeze Gate，禁止启动 Exp02。 | Exp01 |
| 2026-08-12 | D012 | Dataset Freeze Gate PASS 后本轮允许哪些 Exp02 工作？ | 明确授权 Exp02.0--02.2；只用 train/val，test 禁止评估或调参；Exp02.3 和后续方法禁止执行。 | Exp02 |
| 2026-08-21 | D013 | Exp12 是否改变已关闭的最终实验结论？ | 不改变。Exp12 是 `PROJECT_COMPLETE` 后的独立表征学习扩展，不能替换 Exp11 Baseline、重开模型选择或使用 TEST 调参。 | Exp12 |
| 2026-08-21 | D014 | Exp12 普通 SimSiam 何时可进入下游？ | 先通过 TRAIN-only、结构、optimizer/gradient、单步与正式 checkpoint 参数更新 Gate；`changed_ratio == 0` 时标记 `INVALID_BY_BACKBONE_NO_UPDATE` 并停止。 | Exp12.0–12.3 |
| 2026-08-21 | D015 | Exp12.0 现场结构探针是否通过，加载后能否直接训练？ | 结构与 TRAIN-only Gate PASS，但加载时 encoder `requires_grad=True` 为 `0/120`；Exp12.1-S 必须显式解冻并在任何 step 前通过 optimizer/gradient/update Gate。 | Exp12.0–12.1 |
| 2026-08-22 | D016 | Exp12.1-S 是否证明 YOLO backbone 被真实更新，能否自动进入正式 SSL？ | 工程 Gate PASS：单步和一整个 epoch 均改变 `120/120` encoder 参数，梯度、collapse、checkpoint/export reload 和 TRAIN-only Gate 全 PASS；但只授权到 smoke，正式 SSL 仍需单独确认。 | Exp12.1-S–12.2 |
| 2026-08-22 | D017 | Exp12.1/12.2 是否完成且可否自动进入下游？ | 100-epoch TRAIN-only SimSiam 与正式参数审计 PASS：120/120 encoder 参数变化且 checkpoint/export round-trip 120/120；但这只证明真实域适配更新，不能证明 mAP 收益，Exp12.3 Route A/B 仍需单独确认。 | Exp12.1–12.3 |
| 2026-08-22 | D018 | Exp12.3 公平下游是否支持普通 SimSiam 的表征收益结论？ | 工程和公平性 Gate PASS；单 seed42 冻结 VAL 上 Route B Mask mAP50-95 比 Route A 高 0.022916，但 precision 上升、recall 下降。结论限定为单种子 VAL 正信号，不替换最终 Baseline、不推断 TEST 泛化、不自动进入 Multi-scale。 | Exp12.3 |
