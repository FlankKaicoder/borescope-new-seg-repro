# Borescope Multi-class Segmentation Reproduction

本工程用于新孔探多类别 polygon 数据集的可审计方法重建与研究性对比。

Exp00.0–00.5 已完成基础审计与阻塞项审查材料生成。当前状态为 **Data Gate STOP / Environment Gate PASS / Exp01 NOT ALLOWED**：24 张无 JSON 图和 22 个近重复跨类冲突组仍等待人工语义决定。

当前实测训练拓扑为单张约 22GB RTX 2080 Ti；初始“双 11GB GPU”规划已被 Exp00 实测覆盖。项目 `.venv` 已完成依赖与 CUDA/Ultralytics/OpenCV smoke，但这不代表允许训练。

原始数据目录 `/root/autodl-tmp/损伤训练数据集` 视为只读，任何派生产物不得写回。

开始工作前阅读 `AGENTS.md`、`docs/PROJECT_STATE.md`、`docs/DECISION_LOG.md`、`docs/HISTORICAL_METHODS.md`。当前阶段交接见 `docs/handoffs/EXP00_TO_EXP01.md`。
