# Exp14-A3-D2 修订 AMP 协议与有界 Smoke

**PASS_BOUNDED_SMOKE / EXP14_A3_D2_PENDING_CHATGPT_FORMAL_TRAIN_GATE**。两组各完成 2 轮、44 批、44 次有效参数更新。未进入第 3 轮，未执行 VAL/TEST 评价或正式训练。

## 冻结身份与数值协议

分支 phase/exp14-pseudonormal-region，起始本地与 GitHub HEAD 为 78ff69dfe03adb3111f5e67505cc95cc6c081336。服务器在独立 checkout 从同一 HEAD 执行本阶段适配代码。运行代码以 SHA256 固定。Git 提交和远端核验见交付回执。

两组各自从同一 COCO 权重初始化，初始模型状态 SHA256 与 A3/D1 一致。保留 A2 的模型、Loss、数据、100 轮计划、seed42、imgsz640、batch32、AdamW、nbs32、accumulate1、学习率及增强。1,373 个 TRAIN 载荷文件在运行前后通过集合、长度和 SHA256 校验。3 条 UI 风险继续保留，历史 TRUSTED_PAIR_PASS=0 保持有效。

AutoDL 使用 RTX 3080 Ti 12GB、Python 3.12.3、PyTorch 2.8.0+cu128、Torchvision 0.23.0+cu128、Ultralytics 8.4.117。未升级软件。两组 GPU 峰值已分配约 4.80 和 4.84 GiB。

D2 使用 CUDA AMP 和 FP32 主参数，GradScaler 仅将初始 Scale 从 65,536 改为 1。保留 PyTorch 原生 unscale、overflow 检测、step 跳步和 update。对检测到溢出的批次先允许原生跳步和 backoff，记录后停止匹配验收。有限分支沿用梯度范数裁剪 10 和 AdamW。逐批分别记录前向、反向、缩放和 unscale 后梯度、原生 step、有效参数变化和 Scale。 [PyTorch 2.8.0 GradScaler](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/amp/grad_scaler.py)

## 实际采样与优化

| 实测项目 | E0 | E1 |
| --- | ---: | ---: |
| 完整轮数 / 消费 batch | 2 / 44 | 2 / 44 |
| 每轮样本 / 尾批 | 684 / 12 | 684 / 12 |
| 每轮原图 / Aligned | 684 / 0 | 668 / 16 |
| 每轮增强前正实例 / 显式空标签 | 1,284 / 0 | 1,266 / 16 |
| 原生 AdamW step / 有效更新 | 44 / 44 | 44 / 44 |
| 溢出 / 跳步 | 0 / 0 | 0 / 0 |
| Scale 轨迹 | 1 → 1，44 次 | 1 → 1，44 次 |
| 有限 loss 范围 | 183.205–503.923 | 173.309–502.360 |
| unscale 后、裁剪前梯度范数 | 261.09–5,352.00 | 141.90–19,921.72 |

实际 SegmentationTrainer 的两组槽序、逐来源曝光、七类正标签、22 个批次的批界和 LR 轨迹均与冻结计划一致。每轮 21 个满批和 1 个尾批完整消费。第 2 轮换用冻结的第 2 轮顺序并重新启动 worker，记录中两轮 worker PID 不重叠，无跨轮旧样本和额外取图。Mosaic、MixUp、CutMix、Copy-Paste 在实际配置中均为 0。未写训练权重。

## 数值证据与限制

88 批前向和 loss 均有限。所有缩放梯度和 unscale 后梯度均有限，每批 290 个参数梯度张量。全部主参数在每步后保持 FP32 且有限，至少一个非零学习率参数组在每批实际改变。bias 组 44/44 次有变化，weight 和 BN 各 43/43 次有变化，首批的 weight 和 BN 学习率为零。两组最终参数 SHA256 均不同于初始参数。

D2 E0 首批总 loss 390.69805908203125 及各分量与 A3/D1 逐值一致。该批在 Scale 1 下完成有限反向和有效更新，与 D1 的首批 Scale 溢出判断一致。A3 原始失败和 D1 五次回放保留。

零梯度比例的中位数在 E0 为约 0.000158%，E1 为约 0.000176%。E0 最大约 0.00316%。E1 第 2 轮第 13 批有一次 9.60% 的尖峰，其余 E1 批次最大约 0.00190%。该批 32 个槽均为原图，无额外槽和空标签。梯度范数为 544.69，三组非零 LR 参数均变化，Scale 保持 1。日志仅有总体零值计数，缺少该状态下的逐层梯度和 FP32 对照，无法归因于 FP16 下溢。将该指标带入下一道 Gate 监控。

## 正式训练 Gate 建议

本轮受测路径无新增必须修复的 Trainer 行为。可将 AMP init_scale=1 提交为正式协议候选。保持原生 growth_factor=2、growth_interval=2000、backoff_factor=0.5。CPU 合成测试验证了增长边界。若连续 2,000 次更新均无跳步，Scale 将在第 91 轮第 20 批更新后从 1 增至 2，余下 200 批会开始暴露于新 Scale。两轮 Smoke 未覆盖这段真实模型轨迹。

建议正式 Gate 明确两组继续逐批记录 Scale、跳步、真实更新和零梯度比例，在首次 Scale 增长前后保留数值证据。若任一组发生跳步，停止配对预算验收，报告实际有效更新数，等待配对处理决定。固定计划仍为 2,200 批，实际完成预算以 2,200 次有效更新分别核验。当前仅完成有界工程验证，正式入口和评价权限仍待下一道 Gate。

5 项 D2 CPU 合成测试、9 项 A3/D1 CPU 测试和 38 项既有回归通过。两组实际 GPU Smoke 与独立日志复核通过。 [脱敏汇总](../../results/exp14_a3_d2/revised_smoke_summary.json) · [修订适配器](../../tools/exp14/revised_amp_smoke.py) · [独立审计工具](../../tools/exp14/audit_revised_amp_smoke.py)
