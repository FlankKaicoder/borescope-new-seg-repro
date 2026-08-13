# Exp02 NaN root-cause review handoff

## Decision required

Exp02.2a 已完成，实验归类为 `Case C / MODEL_FORWARD_NUMERICAL_INSTABILITY`。**Exp02 Baseline Gate 仍为 STOP，需要人工决定是否对已定位的早期 training-validation AMP overflow 建立严格豁免。**

## Frozen evidence

- 原 Exp02.1：best epoch 99，VAL mask mAP50-95 `0.29898123412927496`，`best.pt` SHA256 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`。
- `best.pt` 重新加载 PASS，561 个 state tensors 全 finite；原权重与 Exp02.2 结论均未覆盖。
- 原始与短复现均为 epoch1--5 四项 val loss NaN，epoch6 恢复；短复现为 `YES`。
- epoch1--5 每个 AMP val batch 的 raw prediction/loss 均 non-finite；相同 checkpoint/batch 的 FP32 全 finite；epoch0/6 两路均 finite。
- 首发点为 `model.10.m.0.attn` 的 FP16 `qk` matmul overflow，然后 softmax NaN；输入、GT、权重与 q/k 在此前均 finite。
- `test_accessed=false`；未跑 Exp02.3/Exp03，未重跑完整 100 epochs，未修改 dataset/model/loss/site-packages。

## Recommended review outcome

不建议立即 controlled full rerun：6-epoch 复现已经稳定重现原模式并定位到具体算子。建议人工评审是否将 Gate 改为：

```text
train loss must be finite
final checkpoint must be loadable and all state tensors finite
independent VAL metrics must be finite
FP32 validation-loss audit must be finite
documented early training-validation AMP forward overflow may be accepted only by explicit human waiver
```

诊断本身不把 Gate 改 PASS。若人工后续 PASS，下一优先建议是 Exp03 low-confidence threshold sweep，不是高分辨率训练。

## Source synchronization notice

服务器中 `ROADMAP.md` 和 `CHANGELOG.md` 已更新到 Exp02.2a complete / Baseline Gate STOP。如 Source 中仍显示 Exp01 或 Data Gate STOP，请重新上传服务器仓库中的最新版本。

详细数值证据见 `docs/exp02_2a_early_val_nan_root_cause.md` 与 `results/diagnostics/exp02_2a_early_val_nan_20260812T145640Z/diagnostic_summary.json`。
