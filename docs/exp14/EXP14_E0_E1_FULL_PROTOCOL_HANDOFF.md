# Exp14 E0/E1-FULL 执行准备交接

`EXP14_FORMAL_PROTOCOL_READY_PENDING_CHATGPT_GATE`。本轮完成协议与双臂执行身份冻结，正式训练及模型评价尚未启动。首轮确定为 seed 42、同一 COCO 初始化、100 轮、640、batch 32、AdamW、原生 AMP Scale 1，无梯度累积。E0 与 E1-FULL 预算分别为 2,100 和 3,300 个有效更新，两组原图及原监督分别为 66,800 张次和 126,600 个增强前实例次。

## 当前实例上的零更新验收

- Windows 工作树与 GitHub 基线为 `5442a0cb28548b76a64dcdc568476e3660ce91c5`。AutoDL 连接端口发生变更，仍用已有固定主机密钥严格校验。当前 GPU 为 RTX 3080 Ti 12 GB，Python、PyTorch、Torchvision、CUDA 和 Ultralytics 与 A11-1 的运行环境一致。
- 两臂运行了真正的正式入口默认 `PREP_ONLY`。A10 的 2,094 个 TRAIN 图像和标签、2,101 个只读运行文件、冻结划分身份、COCO 权重及代码清单均核验。A10 内容 SHA256 为 `6c899ab76fab35832ae38cf8fd67bd0a8a5439ee5bafd48670ce2c37df3321f9`。本轮开始和结束的部署校验报告完全一致。
- 独立构建两臂正式 Trainer 的原生模型、优化器和 DataLoader，并禁用反向及步进调用。各臂以 seed 42 复放同一原生增强 epoch，顺序、图像、标签、框、实例索引与掩膜的合并字节哈希一致。下一轮哈希变化，完整成员和尾批仍匹配。E0 每轮 21 批、尾批 28，E1 每轮 33 批、尾批 23。原生 warmup 读数为 63 和 99。两臂初始参数与 A11-1 一致，模型状态结束时未变。
- 另用 A11-0 的原生读取审计回放 215 个 TRAIN 批次。31 项 Windows 静态／合成与既有数据审计测试、17 项 AutoDL 正式协议相关测试通过。本轮实际参数更新、反向传播、VAL／TEST 图像与标签文件读取、VAL／TEST 推理均为 0。

## 下一道 Gate 需审的边界

1. 审核等 epoch 的主要比较、100 轮末 FP32 主模型选择及独立冻结 VAL 入口。首轮只能给出单种子比较，E0-UPDATE 与额外配对种子预注册为后续对照。
2. 正式训练应在同一张 GPU 上顺序执行两臂。每步监控原生有效更新、LR、冻结曝光和数值状态。第 2,000 次更新后两臂预期都进入 Scale 2。A11-1 的实测仅覆盖 warmup 前段，完整 warmup、Scale 增长后的本数据实际轨迹和长时间运行仍待正式阶段核验。
3. 每 5 轮保留原子发布的恢复状态，终点保存固定 `last.pt`，严格重载七类主权重、EMA、AdamW 和 GradScaler。在约 11.2 GB 可用的数据盘上，各臂预计保留约 0.9 GB 的状态文件及日志。常规失败触发硬 Gate 后先保全报告，外部中断可用上一完整 checkpoint 及文件哈希另建续跑目录。恢复过程使用相同轮种子、优化器、调度器、Scaler 和 RNG。

## 入口与证据

训练入口为 `python -m tools.exp14.a11_full_train`。必须传入 `--protocol`、`--identity`、`--release`、`--deployment-manifest`、`--weight`、`--output` 和 `--arm e0|e1`。默认仅验证，不需运行训练。正式启动时使用 `--apply-gate` 指向获准执行的私有文件，它绑定配对范围、协议哈希和运行身份哈希。续跑还要绑定上一日志及 checkpoint 的实际哈希。VAL 通过 `python -m tools.exp14.a11_full_eval` 独立调用，执行记录绑定两份通过训练验收的第 100 轮 checkpoint，评估器只允许冻结 VAL。

AutoDL 独立准备路径为 `/root/autodl-tmp/exp14_full_protocol_20261115`，沿用 A11-0 的只读发布与冻结 COCO 权重文件。私有运行回执保存在本地证据目录并同步至准备目录的 `reports`。实际执行代码和最终 GitHub HEAD 随回执归档。本轮无训练授权文件，也未创建正式实验结果。

[正式协议](EXP14_E0_E1_FULL_FORMAL_PROTOCOL.md) · [冻结 JSON](../../results/exp14_e0_e1_full_protocol/protocol.json) · [执行身份](../../results/exp14_e0_e1_full_protocol/execution_identity.json) · [脱敏汇总](../../results/exp14_e0_e1_full_protocol/summary.json)
