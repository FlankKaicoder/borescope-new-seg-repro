# Exp13.0-B VLM Semantic Validation

## 实验目的

建立 VLM 语义验证阶段的输入、结构化输出和人工审核记录框架，评估真实七类孔探缺陷图像是否能够被稳定映射为后续生成 prompt 可用的缺陷属性。当前文档只定义验证准备流程，不包含 VLM 实际推理结果。

## 输入样本

- 输入为 Exp13.0 已建立的 40 张 TRAIN-only 样本。
- Crack：20 张。
- corrosion：20 张。
- 每张图片均保留真实类别和样本 ID，用于后续人工核对。

## 数据约束

- 仅使用 TRAIN split。
- 不访问 VAL 或 TEST split。
- 不修改原始数据、标注或数据划分。
- 当前阶段不调用外部 API，不下载模型，不使用 GPU。

## 输出格式

VLM 输出必须遵循 `results/exp13_vlm_semantic_validation/metadata/vlm_output_template.json` 的固定 JSON 字段：类别、几何形态、边界、纹理、颜色、尺度、方向、表面、光照、工业场景、generation hint 和 confidence。

人工审核记录使用 `review/semantic_review_template.csv`，至少评价类别是否正确、属性描述质量以及 generation hint 是否有用。

## 阶段边界

本阶段不生成图片，不进入 Diffusion、ControlNet、SSL 或 YOLO 训练。后续是否开展实际 VLM 推理和生成式实验，必须经过独立 Gate 审核。
