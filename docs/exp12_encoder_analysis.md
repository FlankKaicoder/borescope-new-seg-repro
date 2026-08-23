# Exp12.0 SimSiam encoder 结构分析与执行计划

状态：**EXP12.2 PASS / WAITING_EXP12.3_AUTHORIZATION**
日期：`2026-08-21`
当前仓库基线：`6048421097afd0b791c96ed148c5aeb89f31e946`
最终项目结论：**保持 Exp11 的 YOLO11n-seg Baseline 结论不变**。

## 1. 任务边界

Exp12 是 `PROJECT_COMPLETE` 之后的独立表征学习扩展，不是 Exp00–Exp11 的续跑，不参与最终模型重选，也不回写已有 checkpoint、结果目录或结论。

本轮只完成 encoder 检查、SimSiam 设计、参数更新审计设计和分阶段计划。没有创建 optimizer、没有 backward、没有训练、没有访问 VAL/TEST 图像。

气孔总结和两张流程图仅作为历史设计参考。附件中的旧路径、实验号、超参数、结论和操作指令均不作为本项目命令；Multi-scale Local Change Localization 推迟到普通 SimSiam 通过全部 Gate 之后。

## 2. 已确认的 YOLO11n-seg backbone

证据来自冻结环境 `ultralytics==8.4.117`、现有 Exp09 源码和 Exp09.2a 逐参数审计，而不是复制气孔项目代码。

| 项目 | Exp12.0 判定 |
|---|---|
| 基础权重 | `/root/autodl-tmp/borescope-new-seg-repro/weights/yolo11n-seg.pt` |
| Backbone 边界 | `model.model[0:11]`，即层 `0–10`；Neck 从层 `11` 开始 |
| 模块序列 | `Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF, C2PSA` |
| 输出步长 | `32` |
| 512 输入末端特征 | 预期 `[B, 256, 16, 16]`；必须由 Exp12.0 运行时探针复核 |
| 全局池化后维度 | `256` |
| 可训练参数张量 | 既有实测 `120` |
| 参数元素 | 既有实测 `1,365,472` |
| 完整 state 项 | 既有实测 `240`，包含 parameter 与 BN buffer |

| 层 | 模块 | 输出通道 | 512 输入空间尺寸 |
|---:|---|---:|---:|
| 0 | Conv | 16 | 256×256 |
| 1 | Conv | 32 | 128×128 |
| 2 | C3k2 | 64 | 128×128 |
| 3 | Conv | 64 | 64×64 |
| 4 | C3k2 | 128 | 64×64 |
| 5 | Conv | 128 | 32×32 |
| 6 | C3k2 | 128 | 32×32 |
| 7 | Conv | 256 | 16×16 |
| 8 | C3k2 | 256 | 16×16 |
| 9 | SPPF | 256 | 16×16 |
| 10 | C2PSA | 256 | 16×16 |

精确 class、`from`、输入/输出 shape 将由 `scripts/exp12_0_encoder_analysis.sh` 写入 `encoder_structure.json`。如果运行时事实不一致，`EXP12_ENCODER_STRUCTURE_GATE` 停止。

### Exp09 根因

旧 Exp09 同样使用层 0–10，但深拷贝后没有显式解冻。最终 checkpoint 相对 COCO 为：trainable parameter `0/120` 改变，BN buffer `120/120` 改变；transfer key/shape 和 round-trip 后来已证明正确。

因此失败点是上游 backbone 没被 optimizer 更新，而不是注入机制。Exp12 不允许用 loss 下降、BN 变化或 projector 更新代替 backbone 参数更新证据。

## 3. SimSiam encoder 设计

```text
train/images 中一张图像
        ├─ augmentation t1 → x1 ─┐
        └─ augmentation t2 → x2 ─┤
                                  ↓（共享权重）
                      YOLO11n-seg layers 0–10
                                  ↓
                     AdaptiveAvgPool2d(1)
                                  ↓
                           feature f: 256
                                  ↓
                  projector: 256→2048→2048→2048
                                  ↓
                         embedding z: 2048
                                  ↓
                    predictor: 2048→512→2048
                                  ↓
                    prediction p: 2048
```

```text
L = 0.5 × [D(p1, stopgrad(z2)) + D(p2, stopgrad(z1))]
D(p,z) = -mean(cosine_similarity(p,z))
```

设计约束：

1. 两个分支共享同一个 encoder/projector/predictor。
2. stop-gradient 只作用于对侧 `z`；不得 detach encoder feature、projector 输入或 prediction。
3. encoder 深拷贝后必须显式对层 0–10 的全部参数调用 `requires_grad_(True)`。
4. optimizer 必须按 object identity 完整包含全部 encoder 参数且无重复；projector/predictor 单独统计。
5. 末端特征做 adaptive average pooling；不接 Neck、Segment Head，不实现 change mask 或 multi-scale alignment。
6. 增强与 optimizer/scheduler 在 smoke 前一次性冻结；短程 smoke 的 scheduler horizon 与正式周期解耦。

## 4. TRAIN-only 数据 Gate

用户要求的 TRAIN 图像在当前冻结数据集中的真实目录是 `/root/autodl-tmp/borescope-new-seg-data/v1/images/train`。

DataLoader 建立前必须：

- 冻结 `images/train` 的 668 个逻辑文件必须全部是符号链接；
- 与冻结 split manifest 的 TRAIN 集合 exact-set 核对，预期 668 张；
- 每个链接的文件名必须与 manifest 一致，物理目标必须位于只读原始根 `/root/autodl-tmp/损伤训练数据集`；
- 不因物理目标位于只读原始根而把合法 TRAIN symlink 误判为 split 越界；
- 记录 `ssl_train_images=668`、`val_images_seen=0`、`test_images_seen=0`；
- 不扫描、解码或缓存 `images/val`、`images/test`；
- 不把 VAL/TEST 用于 SSL、阈值、hard mining、epoch/checkpoint 或增强选择。

任何越界、数量/集合不一致或 split 混入均触发 `EXP12_DATA_LEAKAGE_GATE`。

## 5. 参数更新审计方案

### 5.1 训练前后快照

对 `encoder.named_parameters()` 每项保存 `name`、`shape`、`dtype`、`numel`、`requires_grad`、原 dtype SHA256、float64 mean/population std 和 finite。BN running statistics/counter 用 `named_buffers()` 写独立报告，不混入 parameter 更新率。

训练后在内存模型、`last.pt` 重载模型、导出 backbone 三处与 before 对比，并为每个 parameter 保存 before/after hash、mean/std、exact equality、max/mean absolute delta、L2/relative L2 和 changed element count。

### 5.2 optimizer、gradient 与单步 Gate

正式长训练前必须证明：

1. encoder 全参数 `requires_grad=True`；
2. 全部 encoder 参数在 optimizer 中恰好出现一次；
3. loss finite 且有 `grad_fn`；
4. backward 后分别统计 grad 存在、finite、non-zero 的张量和元素；
5. `optimizer.step()` 后至少一个 encoder parameter 精确变化；
6. projector/predictor 变化不能代替 encoder 变化。

第 5 条失败时立即标记 `INVALID_BY_BACKBONE_NO_UPDATE`，禁止 1 epoch smoke、正式 SSL 和下游。

`parameter_update_report.json` 至少包含：

```json
{
  "changed_parameter_count": 0,
  "total_parameter_count": 0,
  "changed_ratio": 0.0,
  "changed_parameter_element_count": 0,
  "total_parameter_element_count": 0,
  "buffer_changed_count": 0,
  "status": "PASS_OR_INVALID"
}
```

主 Gate 是 `changed_ratio > 0`。同时报告分层覆盖率与 relative L2，但不把“变化越大”解释为表示越好。原 dtype hash 与统一 float32 数值比较分开，避免重复 Exp09 的 native FP16 checkpoint 误判。

## 6. Collapse 监控

每个 epoch 对两个视图拼接后记录：

- loss；
- global-pooled encoder feature 的全元素 mean；
- feature 按 batch 逐维 population std 的维度均值；
- projector embedding 按 batch 逐维 population variance 的维度均值；
- learning rate 与 finite 标志。

预注册告警：`feature_std < 1e-4` 或 `embedding_variance < 1e-6` 连续 5 个 epoch，标记 `COLLAPSE_SUSPECTED` 并停止；任意 NaN/Inf 立即 `HARD_GATE`。保留全部逐 epoch 原始值。

## 7. 分阶段计划

| 阶段 | 内容 | 进入下一阶段条件 | 计划输出 |
|---|---|---|---|
| Exp12.0 | 无训练结构/数据/初始参数审计 | encoder、参数/state 数和 TRAIN-only 全 PASS | `results/ssl/exp12_0_encoder_analysis/` |
| Exp12.1-S | 单步 + 真实数据 1 epoch smoke | **PASS**：单步与整 epoch 均为 120/120 参数变化 | `results/exp12_simsiam_basic/smoke_20260822T063541Z/` |
| Exp12.1 | 普通 SimSiam 固定 100 epochs | **PASS**：100/100 finite，无 collapse | `results/exp12_simsiam_basic/formal_20260822T071127Z/` |
| Exp12.2 | 参数和 checkpoint 审计 | **PASS**：120/120 变化且内存/reload/export 一致 | `parameter_update_report.json` |
| Exp12.3 | Route A/B 下游微调 | Exp12.2 PASS 且再次获得确认 | 独立 A/B 目录和公平性报告 |

当前停止点：**Exp12.1 与 Exp12.2 已 PASS；只证明 backbone 真实更新且训练未坍塌，Exp12.3 下游公平比较仍等待单独确认。**

## 8. 下游公平比较预注册

```text
Route A: official COCO YOLO11n-seg → supervised fine-tuning
Route B: official COCO YOLO11n-seg
         → TRAIN-only SimSiam domain adaptation
         → inject layers 0–10 only
         → supervised fine-tuning
```

两条路线固定相同 split、epoch budget、batch、optimizer、scheduler、augmentation、seed 和 evaluator。Route B 必须 strict key/shape match；Neck/Segment Head 相对 Route A 初始化改变数为 0；手工注入后检查一次，Trainer 构造后再检查一次。比较使用冻结 VAL 口径；TEST 不参与训练、早停、阈值、模型或 checkpoint 选择。

Exp12 只能作为独立扩展证据，不替换 Exp11 baseline 或既有 TEST 结论。普通 SimSiam 未通过前不实现 Multi-scale Local Change Localization。

## 9. 本轮产物

```text
tools/ssl/exp12_encoder_analysis.py
scripts/exp12_0_encoder_analysis.sh
docs/exp12_encoder_analysis.md
docs/handoffs/EXP12_0_DESIGN_REVIEW.md
```

`results/ssl/exp12_0_encoder_analysis/`、Exp12.1-S smoke 和 `formal_20260822T071127Z` 均已生成并通过审计；正式 final checkpoint 与 backbone export 保留服务器，轻量证据已完成本地 SHA256 核验。

运行时结构、smoke 与正式结果分别见 `docs/exp12_0_runtime_probe.md`、`docs/exp12_1s_smoke.md` 和 `docs/exp12_1_formal_and_exp12_2_audit.md`。
