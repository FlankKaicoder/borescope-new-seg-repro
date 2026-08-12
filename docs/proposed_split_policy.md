# Proposed split policy（候选，不是最终 split）

当前不生成 `train/val/test`，不生成 `split_manifest.csv`。本文只定义 Data Gate 通过后 Exp01.1 应遵守的候选策略。

## Group policy

| 情形 | 候选策略 |
|---|---|
| SHA256 exact duplicate | 必须同组；若保留多份，整体进入同一 split |
| 高置信 near duplicate | 必须使用 Exp00.3 连通分量作为同一 `group_id` |
| 连续采集/视觉高度相似 | 优先同组；若取得视频/工件/发动机/采集批次 ID，以真实来源组为上位约束 |
| 标签冲突且未确认 | Data Gate STOP，不进入最终 split |
| 无 JSON 且语义未确认 | Data Gate STOP，不进入最终 split |

## Future split objective

- 固定 seed=42，目标 70/15/15；
- group-aware + multilabel-stratified；
- 稀有类 Tip curl、Tears 尽量覆盖 val/test；
- 同时审计 JSON version、PNG/JPG、图像尺寸和类别分布，避免批次效应；
- test 冻结后不参与阈值、hard mining、SSL 或模型选择；
- 输出唯一 `split_manifest.csv` 与 SHA256。

## Required manifest fields

`image_path,json_path,stem,split,group_id,sha256,labels_present,instance_count,json_version,image_suffix,image_width,image_height,source_group,review_status`

## Group construction order

1. 真实来源组（若获得）；
2. exact duplicate group；
3. high-confidence near-duplicate connected component；
4. 文件序列/视觉相似的保守扩展组；
5. 单图组。

组约束合并时取传递闭包，禁止把同一连通组拆到不同 split。正式 split 仍需用户确认后另行执行。

