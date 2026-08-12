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

