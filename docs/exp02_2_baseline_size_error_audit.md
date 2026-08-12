# Exp02.2 baseline size and error audit

状态：**PASS / VAL-ONLY DIAGNOSTIC COMPLETE**

正式结果：`results/evaluation/exp02_2_size_error_audit_20260812T141548Z/`。模型为 Exp02.1 `best.pt`；test 未访问。首次 `20260812T140456Z` 因 Ultralytics 计数字段 API 差异失败，修复后各轮证据保留，未覆盖。

## Train-only size thresholds

以 train 的 1,266 个 GT polygon 在原图分辨率栅格化，定义 `relative_mask_area = filled polygon pixels / image pixels`：

- Q25 = `0.0020196759259259256`；
- Q50 = `0.007364969135802469`；
- Q75 = `0.029209280303030303`。

阈值固定后应用到 val；val 不参与阈值生成。

## VAL size audit

固定诊断点：`conf=0.25`、NMS IoU `0.70`、mask match IoU `0.50`。

| size | GT | TP | FN | Recall | matched mask IoU mean / median |
|---|---:|---:|---:|---:|---:|
| tiny | 103 | 42 | 61 | 0.40777 | 0.72258 / 0.71860 |
| small | 76 | 42 | 34 | 0.55263 | 0.77849 / 0.80102 |
| medium | 62 | 24 | 38 | 0.38710 | 0.84242 / 0.85794 |
| large | 55 | 15 | 40 | 0.27273 | 0.73415 / 0.78967 |

tiny 并非唯一或最差尺度，性能没有随尺度单调改善；large recall 最低。类别×尺度表显示困难具有类别依赖：Burn-tiny 0.161、corrosion-large 0.105、Crack-medium 0、Material missing small/tiny 0，而 Dent tiny/small 分别达到 0.923/1.000。

## Error audit

固定点共有 296 GT、222 predictions：123 TP、173 FN、99 FP。其中：

- 96 个 GT 有同类候选但 box IoU < 0.50（`low_box_iou`，最大来源）；
- 59 个 GT 无同类候选；
- 12 个定位到同类目标但 mask IoU < 0.50；
- 6 个空间匹配但类别错误；
- prediction 侧为 93 unmatched FP + 6 wrong-class FP。

这只是单一 operating point 的诊断，未做低置信度 sweep，不能宣称哪些 FN 可通过降低阈值恢复。最差 AP50-95 类别为 Crack、Burn、corrosion；不能把差异仅归因于实例数，例如样本很少的 Tip curl/Tears 反而较好。

## Near-duplicate cross-label

| 场景 | GT | errors | wrong class | error rate |
|---|---:|---:|---:|---:|
| singleton | 157 | 99 | 2 | 0.63057 |
| near same-label | 115 | 67 | 3 | 0.58261 |
| near cross-label | 24 | 7 | 1 | 0.29167 |

near-cross-label 没有出现明显错误集中或错类集中；该结论受 24 个 GT 的小样本限制。正式目录含 42 张定性错误图及 Ultralytics confusion matrix、PR/F1/P/R 曲线。

## 对 resolution ablation 的证据判断

当前尺度结果不支持“tiny/small 普遍最差 → 必须升分辨率”的简单假设，主要错误更像类别依赖的定位/检出问题。建议暂不把 Exp02.3 作为已获数据证据支持的下一步；先审查 early-val NaN 与错误样本，再决定是否设计控制 batch/有效批量的分辨率对照。未执行 Exp02.3。
