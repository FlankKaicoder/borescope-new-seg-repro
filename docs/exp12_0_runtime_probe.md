# Exp12.0 server runtime encoder probe

状态：**PASS / NO_TRAINING**
完成时间：`2026-08-21T23:30:04+08:00`
实例：`autodl-container-b67c4c8bad-03c084fc`
GPU：`NVIDIA GeForce RTX 2080 Ti, 22528 MiB`

## Scope

本实验只执行冻结 TRAIN 集合核对、YOLO11n-seg encoder 层级假输入前向和训练前 parameter 快照。没有创建 optimizer，没有 backward，没有训练，没有读取 VAL/TEST 图像。

首次启动在模型加载和输出目录创建前触发 `EXP12_TRAIN_ONLY_RESOLVED_PATH_GATE`。只读诊断确认 `images/train` 的 668 个文件均为冻结构建阶段的符号链接，物理目标位于只读原始数据 `/root/autodl-tmp/损伤训练数据集`。旧 Gate 错把合法 symlink 解析到只读源根视为 split 越界。

修复后 Gate 同时要求：

- 逻辑 `images/train` 文件集合与冻结 manifest 的 668 个 TRAIN 条目 exact-set 相等；
- 668/668 均为 symlink；
- symlink 文件名与目标文件名一致；
- 所有物理目标都位于只读原始数据根；
- 不扫描或解码 `images/val`、`images/test`。

失败尝试没有创建 `results/ssl/exp12_0_encoder_analysis`。修复后的首次实际结构审计成功创建该独立目录，不存在结果覆盖。

## Gate results

| Gate | Result |
|---|---|
| TRAIN manifest exact-set 668 | PASS |
| TRAIN symlink targets under frozen read-only source | PASS |
| Encoder layer count 11 | PASS |
| Module sequence | PASS |
| Encoder output shape | PASS |
| Parameter tensor count 120 | PASS |
| Parameter element count 1,365,472 | PASS |
| State tensor count 240 | PASS |
| All parameters finite | PASS |

运行时结构：

```text
input: [2, 3, 512, 512]
layers: 0–10
modules: Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF, C2PSA
output: [2, 256, 16, 16]
pooled feature dimension: 256
```

最关键的训练前事实：

```text
requires_grad_true_at_framework_load = 0 / 120
```

这与 Exp09 的根因一致。Exp12.1-S 构造 encoder 后必须显式执行 `requires_grad_(True)`，并在 optimizer 创建前后重新审计；否则不得执行任何 optimizer step。

## Artifacts

目录：`results/ssl/exp12_0_encoder_analysis/`

| File | SHA256 |
|---|---|
| `encoder_structure.json` | `891e0a0f8a13cdfca8c7dff94bd873e29df750ab0b5de7069d3df8aa1b2964b3` |
| `parameter_snapshot_before.jsonl` | `40cb81fb6abc99728c49ecf7774d3ea640d11881dea65e1362c62a60613fe2b4` |
| `run.log` | `891e0a0f8a13cdfca8c7dff94bd873e29df750ab0b5de7069d3df8aa1b2964b3` |
| `summary.json` | `4e67d7d88fcc94c4f129290bb8ab0849e1de2697a272d756638f72e3bb2bd0bf` |

服务器与本地下载副本的四项 SHA256 完全一致。

## Stop point

Exp12.0 完成。Exp12.1-S 的单步 optimizer/gradient/update smoke 和真实数据 1 epoch smoke 尚未执行，等待单独确认。Exp11 final baseline 和 TEST 结论不变。
