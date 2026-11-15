# Exp15 J-E1 seed142／242 正式配对训练交接

**`EXP15_JE1_MULTISEED_FORMAL_TRAIN_COMPLETE_PENDING_VAL_GATE`**。四臂各完成 100 轮和 2,100 次有效更新，两个 seed 各有 2,100 批完整配对证据。冻结 VAL 待独立授权。

## 执行身份

训练入口为 `python -m tools.exp15.je0 formal`。代码沿用已审计提交 `826634659a036dcccd8cf4e168beeb75639b6d31`。启动时本地和 GitHub 分支均为 `70a981ecf383300a03a47bd0739f4542ccf8c470`，中间提交只新增两份存储维护文档。

机器授权 `EXP15_JE0_MULTISEED_FORMAL_AUTHORIZATION` 仅绑定 seeds 142／242 和 J0／J1。协议 SHA256 为 `1e114398c58a87337a1addbc4392b4485e651f928e95ca9d8c189af52972a305`，J-E0 预检 SHA256 为 `ebe00b9c0c2a4150837110f737a1a7afeffe3886a391bca6ded59fcd184d1e02`。四臂按 142 J0 → 142 J1 → 242 J0 → 242 J1 启动。每一臂先通过独立读回审计，再启动下一臂。

AutoDL 单张 RTX 3080 Ti 12 GiB，Python 3.12.3、Torch 2.8.0+cu128、CUDA 12.8、Ultralytics 8.4.117。四臂均独立从冻结 COCO 初始化，使用 668 张 TRAIN 图、1,266 个原始实例，640 输入、batch32、AdamW、CUDA AMP、63 个 warmup 位置及全局 L2 裁剪阈值 10。J1 使用固定 8 倍 Polygon SupCon，温度 0.10。前后独立复核两套冻结 TRAIN 图像和标签共 2,672 文件，代码、权重、环境及历史正式终点哈希一致。

## 逐臂验收

| Seed／臂 | Epoch／有效更新 | 原图曝光 | 增强前／后实例曝光 | AMP 跳步 | CUDA 峰值 allocated／reserved（GiB） | 训练审计耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 142 J0 | 100／2,100 | 66,800 | 126,600／123,153 | 0 | 4.696／4.990 | 915.10 |
| 142 J1 | 100／2,100 | 66,800 | 126,600／123,153 | 0 | 4.721／4.992 | 979.82 |
| 242 J0 | 100／2,100 | 66,800 | 126,600／123,287 | 0 | 4.701／5.000 | 945.61 |
| 242 J1 | 100／2,100 | 66,800 | 126,600／123,287 | 0 | 4.725／5.168 | 990.51 |

四臂依次执行及独立核验的墙钟耗时为 65.2 分钟。表中单臂耗时涵盖构建训练器、训练及 TRAIN 复核。CUDA 峰值在训练器构建后重置，覆盖训练与 checkpoint 循环。

两个 seed 各核对全部 100 轮、2,100 批的成员、原图顺序、增强张量摘要、逐图实例曝光、原生优化器组、学习率和 AMP 轨迹。同 seed 的两臂原生初始模型和 BN 均匹配。同一联合 AdamW 调用完成 J1 的主任务和投影头更新。四臂全部 Loss、梯度、FP32 主参数和 AdamW 状态通过逐步有限性检查，每个有效更新均观测到参数变化。各臂 2,100 步均触发裁剪，101 个监测步验证 BN 按各臂原生前向更新。各臂的 GradScaler 在第 2,000 步后从 1 升至 2。没有数值硬 Gate 或重试。

| Seed／J1 | 有效 Anchor 最少／中位每批 | 运行覆盖类别 | 无有效 Anchor 的批次 | SupCon Loss 范围 |
| --- | ---: | ---: | ---: | ---: |
| 142 J1 | 31／57 | 7/7 | 0 | 1.509–4.695 |
| 242 J1 | 34／57 | 7/7 | 0 | 1.481–4.535 |

J1 的四个投影参数张量在 2,100 步均有非零梯度，投影头和 Backbone 参数哈希均发生变化。逐类覆盖、梯度范数和裁剪统计见机器汇总。本轮梯度证据来自训练逐步审计，未追加独立 TRAIN 梯度回放。

## 固定终点与证据

每臂保留首步和每 5 轮的 checkpoint，合计 84 份。固定评价输入为各臂 epoch100 last.pt 内的 FP32 主模型。独立审计复核全部 checkpoint 的文件 SHA256、FP32 主模型、EMA、AdamW 和 GradScaler。终点严格 CPU 重载通过，J1 剥离 Projection Head 后以合成零张量验证原生七类结构。

| Seed／臂 | epoch100 文件 SHA256 | 原生 FP32 主模型 SHA256 |
| --- | --- | --- |
| 142 J0 | `462d40d5b7710f234cc8190c13d6dd7bba11fb6612e968b1f4e5174544997b45` | `69491c7d55d99dca6a3aaf90bfff710792dfa92796dbcdd3947d0da471446b8b` |
| 142 J1 | `0c9db72f4947958a2483bd45e1afbc9704e8cf1a037e64c30ea986a15fc9e77d` | `4ff031db0828392904d500e672a453ff99177a87ddaba575e9567daf5b2b08cf` |
| 242 J0 | `90019223c591a7eb36769c45d68a3661a4abe6dbd75982ddb2b0139ef2cac9c0` | `26a44603a484e44dbed28c99b2e46607a3ff8ca4e19135deeb1f0fb2bef736dc` |
| 242 J1 | `45315a6491cf3e157760495f166a09d3e4597cb51c096f847941a0ae3019b3dd` | `6abe39c56e5668233c15f5a8066631213e39bbc92b82a88b9dbc0616bc54f2e1` |

数据盘从 15.121 GB 可用降至 11.183 GB。开跑前按剩余全部 checkpoint 加 2 GiB 日志及临时空间逐臂复核。四份 epoch100 离机副本及私有审计归档均已对账。服务器保留全部原始训练日志、尝试账本和 checkpoint。脱敏 CSV 的 SHA256 按 Git 中保存的 UTF-8 LF 字节计算。

下一阶段仅将这四份 epoch100 FP32 主模型用于独立授权的冻结 VAL。seed42 重训、VAL 和 TEST 推理均为 0。多种子的性能方向、幅度及稳定性待实测。J-D 入口仍绑定 seed42。后续 VAL 授权需绑定本轮四份终点及冻结指标协议。

[机器汇总](../../results/exp15_je1_formal_seed142_242/summary.json) · [固定终点清单](../../results/exp15_je1_formal_seed142_242/epoch100_endpoints.json) · [seed142 配对账本](../../results/exp15_je1_formal_seed142_242/seed142_matched_batch_ledger.csv) · [seed242 配对账本](../../results/exp15_je1_formal_seed142_242/seed242_matched_batch_ledger.csv)
