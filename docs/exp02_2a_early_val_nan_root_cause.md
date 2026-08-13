# Exp02.2a Early-validation-loss NaN root-cause probe

状态：**DIAGNOSTIC COMPLETE / CASE C / LATER ACCEPTED BY EXPLICIT NUMERICAL WAIVER**

## 实验边界

- 只使用 train 进行必要的 6 epoch 短复现，只使用 val 进行 loss 诊断；`test_accessed=false`。
- 训练内保持 `epochs=100`，通过 callback 在 epoch 6 完成后停止；未运行新的 100-epoch baseline。
- 保持 Exp02.1 的 model/data/imgsz/batch/seed/deterministic/AMP/AdamW 配置，只增加逐 epoch checkpoint 和诊断 callback。
- 未修改 dataset、label、model、criterion 或 site-packages；未使用 epsilon、`nan_to_num`、跳 batch 或删图。
- 未启动 Exp02.3、Exp03 或高分辨率训练。

Formal output: `results/diagnostics/exp02_2a_early_val_nan_20260812T145640Z/`.

## 本机 Ultralytics 8.4.117 调用链

`trainer.validate()` 调用 `BaseValidator.__call__(trainer)`，training validation 选择 `trainer.ema.ema`（否则回退到 `trainer.model`）。每个 batch 的顺序为：

```text
validation batch
-> SegmentationValidator.preprocess
-> EMA model forward under autocast when CUDA && trainer.amp
-> model.loss(batch, raw_predictions)
-> box/seg/cls/dfl per-batch accumulation
-> postprocess/NMS
-> metrics update
-> accumulated loss / len(val dataloader)
```

因此 loss 在 postprocess/NMS 之前计算；逐 batch 累加；任一 batch 的 NaN 都会污染该 epoch 的对应 loss。box/cls/dfl 进入 `v8DetectionLoss.get_assigned_targets_and_loss`，seg 进入 `v8SegmentationLoss.calculate_segmentation_loss -> single_mask_loss`。

已保存实际本机源码路径、SHA256、关键行号及 `inspect.getsource(...)` 输出。根因所在的 `ultralytics/nn/modules/block.py` SHA256 为 `0be3c4ffde5a6ecaff604b060a1440c335890cb1541ae64fbac8fe3704aa5764`。

## 原始模式与短复现

重新解析原 Exp02.1 `results.csv`得到：

- epoch 1--5：`val/box_loss`, `val/seg_loss`, `val/cls_loss`, `val/dfl_loss` 全部 NaN；
- epoch 6：四项 val loss 首次全部恢复 finite，之后 epoch 6--100 保持 finite；
- epoch 1--100 train loss 均 finite，指标字段均 finite。

短复现内部使用 `epochs=100`，实际完成 6 epoch。epoch 1--5 再次四项 val loss 全 NaN，epoch 6 恢复，六个 epoch 的 NaN 字段模式与原实验逐 epoch 完全匹配：`NaN pattern reproduced = YES`。

## Checkpoint SHA256

validation probe 使用进入 validation 前保存的 raw FP32 EMA，避免 Ultralytics checkpoint serialization 的 half/copy 路径混淆诊断。

| Epoch | Validation state SHA256 | Period checkpoint SHA256 |
|---:|---|---|
| 0 | `49dc09fc8fc1a093fafb699aa21979ffaad39bf67ff0a6befba22935e014e90f` | n/a |
| 1 | `95754dab91a8efc28b758c21f645f009bba88e2449d952d92df3b5abb0771480` | `0ac5b95c177f28c78a2030966a0d4c642a6e7800854b2eca07e8000c3e660180` |
| 2 | `e705b672bb4cbe42fb5f3b9c40d302c5e2d969a7f0ce9fd32f96ae4a0be4e080` | `6671fa33d7d6287d3bca92184885270027e0b2d50a0829bf54ce2029813c8ad0` |
| 3 | `8a417b1c15e3db044aa9f4425723eed2307d1bea9503b9e4a4180827cecdd326` | `abe60da49125aa767d896372d30cb12fd5ca2257138c3973d8a92eb6edaf184c` |
| 4 | `07d5955cb98e03f1d8f04ca153587463ad705bbdeccf6305807564c436992646` | `daad4d744278986218b776dcfbf49e3756cda7e8ca7751eb8e9e6d9633825e29` |
| 5 | `6a0ed90f85f2d497b89d7c976f90c0a994bf18655fcd2fe2b1c6db881874f043` | `a92516b4ccfc8879c21eb65fbae8718c42bdef58d935855c3f5c4fbc23dd6ebe` |
| 6 | `ab39480a3638189d74e88bde312131c3c69747eed66b56029623e9744538e090` | `719b9e7938fc131bb6252189e9b0e6e6c7eb10aecd4a4e61daf0c926f4ed21cf` |

## AMP/FP32 逐 batch probe

154 张 val 图在 batch=32 下得到 5 batches。epoch0--6 的相同 checkpoint、相同 batch 分别经过 validator-equivalent AMP 与 FP32，共 70 条记录。

| Epoch | AMP raw/loss non-finite batches | FP32 raw/loss non-finite batches |
|---:|---:|---:|
| 0 | 0/5 | 0/5 |
| 1 | 5/5 | 0/5 |
| 2 | 5/5 | 0/5 |
| 3 | 5/5 | 0/5 |
| 4 | 5/5 | 0/5 |
| 5 | 5/5 | 0/5 |
| 6 | 0/5 | 0/5 |

FP32 路径全部 finite，但 epoch1--5 早期数值非常大，与 AMP 溢出相互印证。

## 首个 non-finite batch 及 GT 审计

- checkpoint: epoch1 raw EMA；precision: validator-equivalent AMP；batch index: 0。
- 32 张图：`10, 1000, 1007, 160, 172, 187, 204, 211, 231, 241, 244, 245, 251, 256, 262, 267, 308, 32, 328, 330, 35, 353, 361, 364, 373, 380, 392, 408, 409, 41, 411, 416`。
- 107 个实例：Burn 18, Crack 4, Dent 2, Material missing 3, corrosion 80。
- image tensor 范围 0--1，全 finite；bbox 全 finite，归一化范围 0.006944--0.991667；classes/batch_idx/masks 全 finite。
- 所有 107 个 mask 非空，未发现 empty/degenerate mask。
- loss 计算前 raw boxes/scores/features/mask coefficients/proto 已经全部 NaN，因此不是 GT 或 criterion 首先产生 NaN。

## 首个 non-finite operation

模块级 hook 将异常限定在 `model.10.m.0.attn` C2PSA Attention。对本机 8.4.117 `Attention.forward` 逐操作复算后：

```text
finite FP16 q, k
-> (q * scale).transpose(-2, -1) @ k
-> 8,974,990 Inf logits / 10,240,000 values
-> softmax produces NaN
-> attended values and subsequent forward outputs become NaN
-> box/seg/cls/dfl all become NaN
```

AMP q/k 本身 finite，`qk_matmul_logits` 只有 12.3536% finite，最大有限值恰为 FP16 上限 65504。同一 checkpoint/batch 的 FP32 matmul 全 finite，logit 最大值约 11,582,104。权重与 attention 输入在此操作前均 finite。

## 根因分类和 Gate 建议

```text
ROOT_CAUSE_CASE = Case C
ROOT_CAUSE_CLASS = MODEL_FORWARD_NUMERICAL_INSTABILITY
```

这是 training-validation AMP 的 forward 数值溢出，不是 loss 内部边界条件，也不能按用户给定的 Case A 归类（Case A 要求 raw predictions otherwise finite）。

原 Exp02.1 `best.pt` SHA256 仍为 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`，Ultralytics 加载 PASS，561 个 state tensors 全 finite，独立 VAL metrics 也 finite，因此最终 checkpoint 在数值上有效。但本诊断不自动豁免 Case C：

- 当前 Baseline Gate 继续 **STOP pending human review**；
- 不建议现在盲目重跑完整 baseline，因为短复现已稳定重现机理；
- 可供人工评审的 Gate 修订方向：train loss、最终 checkpoint、独立 VAL metrics 及 FP32 val-loss 审计必须 finite；对已定位且仅存在于早期 training-validation AMP forward 的溢出，可由人工明确豁免。本轮不自行把 Gate 改 PASS。
- 若人工后续接受 baseline，优先建议 Exp03 low-confidence threshold sweep，但本轮未执行。

后续状态：2026-08-13 用户已显式裁决 `PASS_WITH_NUMERICAL_WAIVER`。上述 STOP 是 Exp02.2a 完成当时的历史状态，NaN 证据和 Case C 分类保持不变。

Exp02.2 的 GT=296, TP=123, FN=173, FP=99 及 size/error 结论全部保留；仍无“目标越小越难”的单调证据。
