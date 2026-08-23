# Exp12.1 formal SimSiam and Exp12.2 backbone update audit

状态：**PASS / WAITING_EXP12.3_AUTHORIZATION**

运行时间：`2026-08-22T15:11:30+08:00` – `2026-08-22T16:14:01+08:00`
实例：`autodl-container-b67c4c8bad-03c084fc`
结果目录：`results/exp12_simsiam_basic/formal_20260822T071127Z/`

## Scope and frozen configuration

Exp12.1 是项目关闭后的独立普通 SimSiam TRAIN-only 域适配实验；Exp12.2 在同一 run 的固定 final epoch 后执行参数、checkpoint 与 backbone export 审计。它们不重开 Exp00–Exp11 模型选择，不改变最终 Baseline 或 TEST 结论。

固定配置：

- YOLO11n-seg encoder layers 0–10；projector 256→2048→2048→2048；predictor 2048→512→2048；
- 512×512、batch 32、100 epochs、seed 42；
- SGD，initial lr 0.00625，momentum 0.9，weight decay 0.0001；
- CosineAnnealingLR，`T_max=100`、`eta_min=0`；AMP 关闭；
- 固定第 100 epoch 为最终权重，不早停、不按 TRAIN loss 选择 best checkpoint；
- 每 10 epochs 保存唯一恢复点，共 9 个；
- 冻结 TRAIN 候选集合 668 张；`drop_last=True`，每 epoch 20 steps / 640 processed samples。

## Exp12.1 training result

100/100 epochs 完成，累计 2,000 optimizer steps。所有 epoch 的 loss、feature mean、feature std、embedding variance 均 finite。

| Metric | Epoch 1 | Epoch 100 | 全程最小值 |
|---|---:|---:|---:|
| loss | -0.0097106763 | -0.9166577935 | -0.9287313342（仅诊断，不用于选权重） |
| feature mean | -0.0645617574 | -0.0842628025 | — |
| feature std | 0.1047977105 | 0.2368886784 | 0.1047977105 |
| embedding variance | 0.9999025553 | 0.9999583215 | 0.9999025553 |

`feature_std < 1e-4` 和 `embedding_variance < 1e-6` 从未发生，连续 collapse 计数始终为 0。该结果证明训练过程没有按预注册标准坍塌；它本身不证明下游表示更优。

## Exp12.2 parameter update audit

正式 run 首个真实 TRAIN batch 再次通过前置 Gate：

- encoder gradient present/finite/non-zero：120/120、120/120、120/120；
- 单步参数张量变化：120/120，`changed_ratio=1.0`；
- 单步变化元素：1,365,192/1,365,472。

第 100 epoch 后：

| Audit | Result |
|---|---:|
| changed parameter tensors | 120/120 |
| changed ratio | 1.0 |
| changed parameter elements | 1,365,471/1,365,472 |
| changed element ratio | 0.9999992677 |
| changed BN buffers | 120/120 |
| all final parameters finite | PASS |
| final checkpoint reload hash match | 120/120 |
| backbone export reload hash match | 120/120 |

因此没有触发 `INVALID_BY_BACKBONE_NO_UPDATE`。与 Exp09 仅 BN buffer 变化不同，Exp12.1 的全部 120 个 trainable backbone parameter tensor 都发生了真实变化。

## Data boundary

- TRAIN manifest exact-set：668/668 PASS；
- 668/668 logical symlink 的物理目标均位于只读原始数据根；
- `val_images_seen=0`；
- `test_images_seen=0`；
- `test_accessed=false`；
- 没有下游训练、阈值调整、hard mining 或 checkpoint 选择。

## Checkpoints and hashes

- 固定 final checkpoint `last.pt`：SHA256 `478a5cd14f850925c8c6ba1a78459d7d93c0076511234a20b4c58f4de0a88c7b`；
- 最终 encoder export `adapted_backbone_final.pt`：SHA256 `3ae6909d1d377ed18c8d532b7fde50cb0d64c092f4137dc035aff5a2a94f580c`；
- 9 个周期恢复点的 SHA256 记录在 `periodic_checkpoints.json`；
- 训练用时 3,751.64 秒；GPU 峰值分配约 5.18 GiB；
- 本地下载了 JSON/JSONL/CSV/log 证据且 SHA256 与服务器逐项一致；约 1 GB checkpoint 保留服务器，不纳入仓库。

## Interpretation and stop point

Exp12.1/12.2 证明：正确显式解冻后，普通 SimSiam 能够在新孔探 TRAIN 域对 YOLO11 backbone 进行真实、有限且未坍塌的参数更新。

尚未证明：该表示能提高分割 mAP 或优于 COCO 初始化。只有公平的 Exp12.3 Route A/B 下游微调才能回答这一问题。Exp12.3、VAL 比较和 Multi-scale Local Change Localization 均等待单独授权；TEST 继续禁止参与。
