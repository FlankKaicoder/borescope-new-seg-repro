# Repository operating rules

- 原始数据 `/root/autodl-tmp/损伤训练数据集` 永远只读；不得重命名、删除、修改或写回派生文件。
- 不覆盖或删除旧实验。每个实验必须有独立编号、脚本、日志、结果目录、Markdown，并更新 registry。
- test 不参与训练、SSL、阈值调参、hard mining 或模型选择。
- 只有 smoke test 通过后才能正式训练。
- 数据语义不确定时不得猜测、自动改标签、自动合并类别或把无 JSON 图片当背景。
- Gate 未 PASS 不进入下一阶段。
- 每次开始实验前阅读 `docs/PROJECT_STATE.md`、`docs/DECISION_LOG.md`、`docs/HISTORICAL_METHODS.md` 和对应前序实验 Markdown。
- 每个实验结束更新 `docs/PROJECT_STATE.md`、`results/experiment_registry.csv`、`docs/experiment_index.md` 和当前 handoff。
- 已被实测推翻的初始假设以当前仓库结果为准；当前 GPU 拓扑为单张约 22GB RTX 2080 Ti。

