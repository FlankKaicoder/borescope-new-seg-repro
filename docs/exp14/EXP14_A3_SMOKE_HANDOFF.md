# Exp14-A3 AutoDL 有界 Smoke

**SMOKE_STOP_PENDING_CHATGPT_REVIEW**。E0 第 1 轮第 1 批在 AdamW 更新之前被非有限 AMP 缩放梯度拦截。两组有效更新均为 0，E1 的有更新阶段未启动。正式 E0/E1 训练前置条件未满足。

## 身份与环境

Exp14 worktree 分支为 phase/exp14-pseudonormal-region，起始本地与远端 HEAD 为 c7c961009947416a0271cede852f38a2bbec39f8。服务器独立 checkout 使用同一基线，另以 SHA256 绑定本阶段实际执行的适配代码。最终提交和远端核验见交付回执。 [GitHub 分支](https://github.com/FlankKaicoder/borescope-new-seg-repro/tree/phase/exp14-pseudonormal-region)

通过现有有效 SSH 配置连接。实测为 RTX 3080 Ti、12,288 MiB，预检起始可用 11,909 MiB，驱动 595.71.05，驱动报告 CUDA 13.2，PyTorch 运行时 CUDA 12.8。独立目录审计时磁盘余量 14.63 GiB。复用 Python 3.12.3、PyTorch 2.8.0+cu128、Torchvision 0.23.0+cu128、Ultralytics 8.4.117、NumPy 2.3.2、Pillow 11.3.0。软件包未变更。

COCO 权重 SHA256 为 55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152。A2 私有 TRAIN 载荷传输 1,373 个白名单文件，服务器逐个校验文件集合、长度和 SHA256，结束后再次校验通过。包括 668 原图、1,266 个原实例、16 张 Aligned 和 16 份显式空标签。三条腐蚀 UI 风险保留。未传输 TEST、人工审核材料或凭证。

## 无更新预检

| 项目 | E0 | E1 |
| --- | ---: | ---: |
| 实际槽数 | 684 | 684 |
| 实际 batch | 22 | 22 |
| 满批 / 尾批 | 21 × 32 / 1 × 12 | 21 × 32 / 1 × 12 |
| 原始正实例曝光量，增强前 | 1,284 | 1,266 |
| 加载的显式空标签 | 0 | 16 |
| 有限前向检查 | 2 | 2 |
| 有效更新 | 0 | 0 |

实际 SegmentationTrainer 从冻结序列构造两组 batch。E0 额外槽重复同源原图索引并保留正标签，E1 额外槽加载 Aligned 和空标签。第 1 轮槽序、每来源曝光、七类正样本、尾批和 worker 返回的 epoch 标签通过检查。两个 worker、prefetch 2、非持久 worker。观察到每个被 Trainer 消费的槽仅请求对应原始索引。Mosaic、MixUp、CutMix、Copy-Paste 均关闭。两组初始模型 SHA256 相同，真实 CUDA autocast 与 GradScaler 启用，主参数为 FP32。训练集以外的评估路径为空，默认训练验证和最终验证均被适配层屏蔽。

## 更新 Smoke 与停止原因

保留 A2 的 100 轮调度配置，在适配层设两轮硬上限。E0 实际只消费第 1 轮首批 32 个样本，batch 与计划一致。AdamW、batch=32、nbs=32，首次 warmup accumulation=1。首批参数组 LR 为 0、0、0.1。模型前向与所有 loss 分量有限，总 loss 约 390.6981。AMP backward 后检查到非有限缩放梯度，拦截发生在原生 optimizer_step 内的 unscale、clip 和 AdamW 更新之前。适配入口进入一次，真实 optimizer.step 调用与有效更新为 0。原始证据可定位到首次缩放反向，尚不足以定位具体参数或确认溢出机制。

本次未完成训练首轮，训练态尾批、两轮实际切换、跨轮预取和完整学习率轨迹尚未验证。无参数更新的预检已经验证首轮真实尾批。未降低 loss scale、调整 batch、关闭 AMP 或重试。无检查点写出，无 VAL/TEST 评价。全部训练相关轨迹标记 DIAGNOSTIC_ONLY。

## 验证与下一道 Gate

4 项 A3 CPU 合成测试及 38 项既有 Exp14 CPU 回归通过。已私有归档服务器转移校验、逐批日志、曝光表、环境及停止记录。公开文件仅含通用适配与审计代码、合成测试和脱敏汇总。 [脱敏汇总](../../results/exp14_a3/smoke_summary.json) · [适配代码](../../tools/exp14/smoke_trainer.py)

下一道 Gate 需先决定是否授权数值定位。若授权，优先保留原首批和配置，定位首次非有限梯度，并由独立对照区分 AMP 缩放与反向算子问题。待原因明确、更新路径和两轮采样通过新的有界 Gate 后，再评估正式训练。本轮已停止。
