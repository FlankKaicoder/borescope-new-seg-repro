# Exp12.1-S SimSiam single-step and one-epoch smoke

状态：**PASS / WAITING_FORMAL_SSL_AUTHORIZATION**
运行时间：`2026-08-22T14:35:44+08:00` – `2026-08-22T14:36:29+08:00`
实例：`autodl-container-b67c4c8bad-03c084fc`
结果目录：`results/exp12_simsiam_basic/smoke_20260822T063541Z/`

## Scope

本阶段只验证普通 SimSiam 是否能真实更新 YOLO11n-seg 层 0–10，并完成一个 TRAIN-only smoke epoch。它不是正式 SSL，不进行下游微调，不访问 VAL/TEST，不改变 Exp00–Exp11 的最终 Baseline 和 TEST 结论。

配置：512×512、batch 32、seed 42、SGD、lr 0.00625、momentum 0.9、weight decay 0.0001、AMP 关闭。一个 epoch 共 20 steps，因 `drop_last=True` 实际处理 640 个样本；DataLoader 的候选集合仍是冻结 TRAIN 的全部 668 张图像。

## Hard Gate results

| Gate | Result |
|---|---|
| 冻结 TRAIN manifest exact-set 668 | PASS |
| TRAIN 668/668 symlink 均指向只读原始数据根 | PASS |
| Exp12.0 初始 120 参数逐项 name/shape/hash 对齐 | PASS |
| 框架加载时 `requires_grad=True` | `0/120`，与 Exp12.0 一致 |
| 显式解冻后 `requires_grad=True` | `120/120` |
| encoder 参数在 optimizer 中恰好出现一次 | `120/120`；duplicate 0 |
| 单个真实 batch gradient present/finite/non-zero | `120/120` / `120/120` / `120/120` |
| 单步 encoder 参数变化 | `120/120`，`changed_ratio=1.0` |
| 一整个 epoch 后 encoder 参数变化 | `120/120`，`changed_ratio=1.0` |
| checkpoint reload 参数 hash | `120/120` |
| backbone export reload 参数 hash | `120/120` |
| VAL/TEST 图像访问 | `0/0` |

单步共有 1,365,449 个参数元素获得非零梯度，1,365,192 个元素发生变化；完整 epoch 后 1,365,467/1,365,472 个参数元素发生变化。BN buffer 变化为 120/120，但 buffer 变化没有被用于替代参数 Gate。

因此本实验没有触发 `INVALID_BY_BACKBONE_NO_UPDATE`。这证明当前显式解冻和 optimizer 构造能够真实改变 YOLO backbone；它不证明表征质量提升，也不构成下游效果结论。

## Collapse and finite monitoring

| Metric | Epoch 1 |
|---|---:|
| loss | -0.0097106763 |
| feature mean | -0.0645617574 |
| feature std | 0.1047977105 |
| embedding variance | 0.9999025553 |

所有指标 finite。`feature_std > 1e-4` 且 `embedding_variance > 1e-6`，smoke 未出现 collapse 信号。一个 epoch 只能完成工程 Gate，不能替代正式训练中的连续五个 epoch collapse 规则。

## Checkpoints and evidence

- 服务器 `last_smoke.pt`：SHA256 `6b07853a02bde6f43d6e7bf7d5acf613bc1fb7d336bc7cc6a5839c624c394718`；标记为 smoke-only，不能用于正式下游。
- 服务器 `adapted_backbone_smoke_only.pt`：SHA256 `15b8007e21077ed59a84be9814faac8e0deddec6a0c5be26ac7d19e0720f9724`；同样禁止冒充正式 SSL 结果。
- 本地只下载 JSON/JSONL/CSV/log 审计证据；约 99 MB 的 smoke checkpoint 保留在服务器，避免把临时权重纳入仓库。
- 下载文件 SHA256 与服务器逐项一致，本地 schema 断言 PASS。

## Stop point

Exp12.1-S 已完成。下一步只能在单独确认后启动 Exp12.1 正式普通 SimSiam TRAIN-only 训练。Exp12.2 正式 checkpoint 参数审计、Exp12.3 Route A/B 下游比较和 Multi-scale Local Change Localization 仍未授权。
