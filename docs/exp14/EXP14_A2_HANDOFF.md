# Exp14-A2 核心数据冻结与 E0/E1 准备

状态为 **FROZEN_DATA_PENDING_CHATGPT_SMOKE_GATE**。A1 数据 Gate 授权冻结本轮数据。GPU smoke 仍需下一道 Gate 和执行授权。起始本地与 GitHub HEAD 为 96fbf897cd4e4283d3cb956a42003fb2250006de，分支 phase/exp14-pseudonormal-region。提交与远端核验见交付回执。 [GitHub 分支](https://github.com/FlankKaicoder/borescope-new-seg-repro/tree/phase/exp14-pseudonormal-region)

## 核心队列和全图支持

从 A1 的 22 个 A 候选和 21 个 B 候选精确取交集，确认 21 个独立 TRAIN 来源、21 个来源组、每来源一变体及 24 个原 Polygon。复核原图、JSON、冻结组别、修复类别、人工 A/B PASS、逐实例 DELETE、整图零目标确认和最终 Aligned 的实际哈希。18 个非共同变体及其历史状态保留在私有排除记录。

全图有效范围由保存的二值 valid_mask 确定，另单独量测有效范围外的精确 RGB 黑填充，并以原图亮度 ≥25 的重叠区域定位可见内容损失。亮度用于显示诊断，逐图对照原图、Aligned 和叠图后决定 HOLD。全队列全图有效率为 74.01%–100%，24 个目标 Polygon 有效支持均为 100%。5 个变体的黑边明显截断原图可见机械部位，标记 TRAINING_HOLD。其余 16 个的全图有效率为 98.42%–100%，无显著的新增机械截断。3 个保留的腐蚀变体在有效范围内去除了原相机 UI，可能成为增强状态的捷径。记录为独立观察，按类解读后续效果。未重配准或替换 Aligned，未改变人工结论。

| 修复目标类 | 保留 | 全图支持 HOLD |
| --- | ---: | ---: |
| Burn | 1 | 2 |
| Crack | 3 | 1 |
| Dent | 1 | 0 |
| Material missing | 2 | 2 |
| Tears | 6 | 0 |
| Tip curl | 0 | 0 |
| corrosion | 3 | 0 |

## 冻结数据及 B 元数据

私有目录含 668 对基线图像与标签副本，以及 16 张字节相同的审核版 Aligned 和 16 份显式零字节标签。基线标签由专业 JSON 用冻结转换规则重建，668 份标签的 SHA256 均与冻结划分中的原标签 SHA256 相符，共 1,266 个实例。文件和来源组逐项校验，仅材料化 TRAIN。

保存全部 24 个原 Polygon 的来源、类别、坐标、Aligned 源网格恒等映射、几何诊断、有效支持掩码及审核和文件哈希。其中 18 个区域对应实际保留的 16 个来源，另 6 个保留追溯信息并标为训练 HOLD。后续 E2/E3 与本次筛查后的核心来源对齐。历史 TRUSTED_PAIR_PASS=0 保持有效。

## E0/E1 匹配采样和训练草案

- 两组每轮均有 668 个共同基线槽和 16 个额外槽。E0 的额外槽使用同来源原图及原标签，E1 在相同位置使用审核版 Aligned 与显式空标签。每个额外来源每轮只出现一个额外槽。
- 已保存 100 轮精确槽序、逐批边界、两组槽位到图像与标签的映射和每来源每轮曝光表。seed42 按轮确定共同顺序。每轮 21 个满 batch 和一个 12 样本尾批，均保留。每组共 68,400 次计划采样、2,200 次计划 optimizer updates。
- 两组固定 YOLO11n-seg、官方 COCO 预训练权重及其历史 SHA256、640、AdamW、100 epochs、seed42。草案固定 batch=32、nbs=32、accumulate=1、lr0=0.01、lrf=0.01、线性日程、预热 3 轮、deterministic 和 AMP，细项见私有 training_protocol.json。此处固定累积为 1 以使实际更新与槽预算可核对。
- Mosaic、MixUp、CutMix、Copy-Paste 均设为 0，避免增强从索引外额外取图。两组共享单图增强设置和随机种子。DataLoader 需用随附 FrozenSlotBatchSampler 执行批次表，避免二次 shuffle、丢弃尾批或跨轮预取改变顺序。E0 的额外槽故意重复同一数据集索引，两个唯一路径清单仅用于构造数据集索引。
- 仅用冻结 VAL 比较最终第 2,200 次更新的 last checkpoint。主指标为 Mask mAP50-95，记录各类 AP、Recall、Precision 和误检，固定 conf=0.001、NMS IoU=0.70。两组共享协议。无 TEST 方法选择。

历史 100 轮 Baseline 作为背景记录。本对照关闭了跨图增强并固定了累积策略，正式方法效应应使用同期 E1−E0。此次冻结采用完整 100 轮预算，未采用 30 轮快速筛查。

## CPU 验证与 Smoke Gate

38 项 CPU 测试通过。Ultralytics 8.4.117 发布包 SHA256 已固定，在 NumPy/PIL 的 CPU 路径执行该包的原版图像与标签解析函数。解析通过 668 份正标签和 16 份显式空标签，保留 1,266 个分割实例。Python 3.14 本地解析环境的 NumPy/Pillow 与历史 AutoDL 运行环境不同，实际 Trainer 和 DataLoader 尚待 Smoke 验证。 [Ultralytics 8.4.117 发布包](https://pypi.org/project/ultralytics/8.4.117/)

数据与计划已具备提交 Smoke Gate 的条件。获准后先在 AutoDL 校验权重和数据哈希，接入实际 DataLoader 并核验两组槽序、尾批、正/空标签、增强抽样、有限 loss 和 optimizer steps。Smoke 通过后再提交正式运行 Gate。本轮未启动 AutoDL、训练、VAL/TEST 或生成 API。 [脱敏汇总](../../results/exp14_a2/freeze_summary.json) · [支持审计](../../tools/exp14/aligned_support.py) · [冻结与采样工具](../../tools/exp14/freeze_core.py)
