# Borescope Multi-class Segmentation Reproduction

本工程用于新孔探多类别 polygon 数据集的可审计方法重建与研究性对比。

Exp00.0–00.5 与 Exp01.0–01.2 已完成。专业 JSON 是权威 GT；24 张无 JSON 图为 `excluded_unpaired`；22 个近重复跨类组原标签不变并作为不可拆 split group。Dataset v1 已冻结于 `/root/autodl-tmp/borescope-new-seg-data/v1`，当前状态为 **Dataset Freeze Gate PASS / Exp02 technically allowed but not started**。

当前实测训练拓扑为单张约 22GB RTX 2080 Ti；初始“双 11GB GPU”规划已被 Exp00 实测覆盖。项目 `.venv` 已完成依赖与 CUDA/Ultralytics/OpenCV smoke，但这不代表允许训练。

原始数据目录 `/root/autodl-tmp/损伤训练数据集` 视为只读，任何派生产物不得写回。

开始工作前阅读 `AGENTS.md`、`docs/PROJECT_STATE.md`、`docs/DECISION_LOG.md`、`docs/HISTORICAL_METHODS.md`。当前阶段交接见 `docs/handoffs/EXP01_TO_EXP02.md`。
