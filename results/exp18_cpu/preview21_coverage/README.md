# EXP18 已保存21批覆盖预览（非完整4200批验收）

当前总状态：**ENGINEERING_PASS_FULL_REPLAY_PENDING**。这里只汇总既有已保存的1个epoch、21批元数据；计划4200批，剩余4179批未纳入。

本次没有新数据回放、测试任务、模型或Torch/NumPy/SciPy导入，没有GPU计算。`maximum_assignment_recomputed=False`：只检查已存闭合置换、合法边及权重；不重新运行最大指派。原summary中的PASS仅指这21批结构审查。

逐类K候选与S实际匹配、全局唯一Polygon/来源组数见 [population_by_class.csv](population_by_class.csv)；这些唯一数量从已存行的全局集合重新统计，没有累计每批unique数。所有数值仅代表本次21批。历史R5权重和交集也只使用对应的21个历史步。

[范围与状态](PREVIEW_SCOPE.json) · [覆盖报告](REPORT.md) · [逐层约束](constraint_funnel_by_class.csv) · [伙伴多样性](partner_diversity_by_class.csv)
