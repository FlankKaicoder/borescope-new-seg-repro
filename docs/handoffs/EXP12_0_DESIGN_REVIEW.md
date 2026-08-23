# Exp12.0 design review handoff

状态：`DESIGN_COMPLETE_NO_TRAINING_WAITING_USER_CONFIRMATION`。

## 已完成

- 阅读 `PROJECT_STATE`、`DECISION_LOG`、`HISTORICAL_METHODS`、Exp09 失败/transfer 修复记录与 Exp11 closure。
- 将气孔 SimSiam 总结和两张流程图仅作为历史参考，未复制旧代码或旧实验配置。
- 确认 encoder 边界为 YOLO11n-seg 层 0–10；既有实测为 120 个 parameter tensor、1,365,472 个 parameter element、240 个 state tensor，末端 256 通道。
- 定义 TRAIN-only 数据 Gate、显式解冻、optimizer identity 覆盖、单步梯度/参数变化 Gate、parameter/buffer 分离审计、collapse 监控和下游双重注入核验。
- 新增无训练结构审计脚本与服务器 launcher；输出目录存在时拒绝覆盖。

## 未执行

- 未连接/附着服务器终端执行运行时结构探针；当前 Codex task 没有附着 app terminal。
- 未创建 optimizer、未 backward、未执行 smoke、未进行 SimSiam 或下游训练。
- 未访问 VAL/TEST 图像，未更改 Exp00–Exp11 结果或最终 baseline 结论。

## 下一步唯一允许动作

用户确认后，先在原服务器运行 `bash scripts/exp12_0_encoder_analysis.sh`。只有 Exp12.0 runtime report PASS，才实现并执行 Exp12.1 的单步与 1 epoch smoke；只有 smoke PASS，才可请求正式 SSL 训练确认。

任何 backbone `changed_ratio == 0` 必须标记 `INVALID_BY_BACKBONE_NO_UPDATE` 并停止，禁止进入下游。
