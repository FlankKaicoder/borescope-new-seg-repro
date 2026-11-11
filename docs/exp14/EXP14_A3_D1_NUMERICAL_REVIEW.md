# Exp14-A3-D1 AMP 首批非有限梯度诊断

**AMP_SCALE_OVERFLOW_LIKELY。** E0 首批按 A3 已留存证据复现。五次独立反向完成，模型参数更新为 0。正式 E0/E1 训练仍停在 Gate。

## 身份和复现边界

分支为 phase/exp14-pseudonormal-region，起始本地、远端和独立服务器 checkout HEAD 为 7fb106581f01b0b34f593eda18fadae49582c5ad。A3 适配器、COCO 权重、冻结 TRAIN 载荷和 A3 停止日志均已核对 SHA256。D1 使用新目录，未更改原 A3 失败记录。

实际复核了 A3 首批的 32 个 Slot ID、数据集索引、图像路径、worker、原类别及 57 个增强前实例。有效训练配置、seed42、worker generator seed、模型初始状态 SHA256 与 A3 一致。R0 的总 loss 和各分量逐值完全相同。A3 当时没有保存预处理张量或其哈希，原批次字节身份无法回溯比对。D1 的预处理张量快照、逐张量 SHA256、RNG 状态和初始化指纹已在私有服务器归档，五次回放均从同一快照和新模型副本开始。各案前后参数 SHA256 一致。

## 五次回放

| 回放 | 精度 | 实际 Scale | 总 Loss | 非有限参数梯度张量 / 元素 | unscale 后 |
| --- | --- | ---: | ---: | ---: | --- |
| R0 | AMP | 65,536 | 390.698059 | 252 / 2,758,518 | 同数非有限 |
| R1 | AMP | 4,096 | 390.698059 | 178 / 1,838,618 | 同数非有限 |
| R2 | AMP | 256 | 390.698059 | 1 / 44 | 同数非有限 |
| R3 | AMP | 1 | 390.698059 | 0 / 0 | 全部有限 |
| R4 | FP32，关闭 autocast 和 TF32 | 无 | 390.722473 | 0 / 0 | 无缩放 |

全部前向预测、总 loss 和 loss 分量有限。四个 AMP 案例的分量一致，依次为 box 1.602265、seg 4.104664、cls 4.860324、dfl 1.642061、sem 0。R4 对应分量为 1.603857、4.106874、4.858298、1.641048、0。每案均有 290 个参数梯度张量，主参数为 FP32。R3 的有限梯度来自独立反向。

R0 首个观察到局部产生非有限梯度的节点为 MvBackward0。其上游梯度有限，返回的 FP16 梯度形状为 [1075200, 16]，含 5,271 个 ±Inf，有限最大值 65,504。该形状与安装版 Ultralytics 的 bbox_decode DFL 投影一致。钩子没有保留前向调用栈，因此精确 Python 调用位置仍属推断。R0 最先到达参数钩子的受影响项为 model.23.proto.cv3.conv.weight，FP32 梯度形状 [32,64,1,1]，2,048 个元素中 1,742 个非有限。R2 的剩余异常集中在 model.0.conv.weight，432 个元素中 44 个为负 Inf。全部名称、dtype、形状和有限元素统计在私有回放记录中。

## GradScaler 与训练计数

A3 在原生 optimizer_step 前检查缩放梯度，先于 GradScaler 的 unscale、溢出检测、skip 和 update。已核对服务器 PyTorch 2.8.0 安装源码。原生 unscale 检测非有限值，step 依检测结果跳过底层优化器调用，update 应用 backoff_factor。D1 用不具备更新实现的保护优化器进行 R0 探针，实测 found_inf=1，底层 step 调用为 0，Scale 从 65,536 降到 32,768。unscale 后 R0 仍有相同数量的非有限元素。本轮没有测试 32,768 是否足以使梯度有限。 [PyTorch 2.8.0 GradScaler 源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/amp/grad_scaler.py)

R0–R2 溢出，R3 和 R4 正常，支持首批的 AMP Scale 溢出归因。证据不覆盖后续 batch 或两轮训练。A2 的 2,200 是计划 batch 数。在 accumulation=1 的协议下仍需区分尝试次数、scaler 跳步和实际成功的参数更新数，跳步会使有效更新少于计划 batch 数。

## 待审批协议建议

建议下一道 Gate 只授权另一次有界 Smoke，保留原数据、batch、AdamW、学习率、Loss 和 AMP autocast，以本批已验证有限的 init_scale=1 作为诊断起点。保留原生 GradScaler 的检测、跳步和 backoff 路径，分别记录缩放状态、非有限梯度、真实 AdamW 调用和有效更新。设定何种持续异常应停止、以及两组遇到跳步时如何保证有效更新预算匹配，均应由 Gate 明确。单批结果不足以冻结新的正式 Scale 或批准 100 轮训练。

5 项 D1 和 4 项 A3 CPU 合成测试在服务器通过，38 项既有 CPU 回归在本地通过。完成 5 次预定反向，未调用底层模型优化器更新、未写训练检查点、未访问 VAL/TEST。 [脱敏结果](../../results/exp14_a3_d1/numerical_summary.json) · [诊断代码](../../tools/exp14/diagnose_amp_first_batch.py)
