# Exp15 J-A 共享 Backbone 区域监督交接

基于 Exp14 的 `83274732f72133cbe9a3abff2ab9a6c66c0e2eea`。执行代码提交为 `ee51cc1baab6fc38fe4c932af88ecff36761742b`，三个 Smoke 源文件的 SHA256 见脱敏回执。本阶段验证机制和训练工程。正式效果比较以后续独立 Gate 为准。

## 方案

本轮只使用七类冻结原始 TRAIN。J0 调用原生 YOLO11n-seg 的主任务 loss。J1 在同一次整图前向中从融合 P3 取特征，叠加真实 Polygon 区域 SupCon。投影头为 `64 → 128 → 128` 的两层 MLP，ReLU 中间层，输出 L2 归一化。两个损失在一个计算图上求和，一次 AMP backward、一次原生 AdamW 更新。训练后的七类分割状态可严格装入原生模型，推理走原生 Segment Head。

| 路线 | 本轮判断 |
| --- | --- |
| 同图 P3 box ROIAlign | Exp08 已验证同图接入的思路。其旧节点是 backbone 层 4，stride 8、128 通道。框采样对细长 Crack 或低填充 Polygon 会平均较多机械背景。固定输出网格仍受原特征分辨率限制。 |
| 独立裁块经过共享 backbone | 可放大极小缺陷，也能独立平衡来源和稀有类。需要第二条图像前向和独立采样，额外核验 BN Buffer、裁块增强、曝光和计算开销。适合在现有 P3 机制的尺寸覆盖确实成为瓶颈时做独立对照。 |
| 同图 P3 Polygon 加权池化，采用 | 原生分割的 Segment 层 23 接收层 16、19、22。层 16 是融合 P3，`[B,64,80,80]`，图像 stride 8。直接复用增强后 `[B,160,160]` 原生 overlap ID raster。将区域的每个 2×2 mask 像素块换算成一个 P3 单元覆盖权重。池化按覆盖率加权，完全在 Polygon 外的格点权重为零。 |

以上实现沿用 Exp14 E0 的冻结输入、原生数据加载和计数、epoch seed/worker RNG、COCO 初始化转移、warmup、AdamW、AMP 和更新审计。Exp08 提供同图 ROI 先例，Exp09 的无参数更新事件要求分别追踪独立辅助梯度、骨干权重和投影头权重。[Exp08](../exp08_kd_fast_repro.md) · [Exp09](../exp09_simsiam_fast_repro.md) · [Exp14 正式协议](../exp14/EXP14_E0_E1_FULL_FORMAL_PROTOCOL.md) · [Ultralytics v8.4.117 模型定义](https://github.com/ultralytics/ultralytics/blob/v8.4.117/ultralytics/cfg/models/11/yolo11-seg.yaml) · [ROIAlign](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html)

样本来源组绑定到固定 split manifest，并核验图像和标签 SHA256。增强后以 batch slot 和原生按面积排序的局部 mask ID 绑定 Polygon、类别、框和图像。每批检查 raster 的空间范围，容许 mask 分辨率下 2 像素栅格误差。被遮挡或栅格化后面积为零的目标从辅助池排除，主任务照常使用。合成 CPU 测试验证错位 mask 会触发停止，以及 Polygon 外 P3 单元的输入梯度为零。P3 本身有感受野，一个或几个有效格点仍可能混合大量机械上下文，空间覆盖只验证取样信号存在。

SupCon 用温度 0.10 和 FP32 logsumexp。跨来源组、同类为正对，跨来源组、异类为负对。同一组的其他样本从该 anchor 的分母排除。Anchor 需同时有正对和负对。先求每类有效 anchor 的均值，再对当前批有 anchor 的类等权平均。现有批次已给出七类信号，因此采用原生批次里的全部有效真实区域。这个设计保持主任务的逐图采样和增强，同时如实保留类别间批次覆盖差异。[SupCon 原论文](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

## TRAIN 预检

四项 CPU 合成测试通过。70 个真实批次共 4,093 个增强后实例曝光，均有原生 raster 和 P3 支持。其中 23 个只覆盖 1 个 P3 单元，227 个覆盖 2–4 个。每批 Polygon 在其矩形框内的填充中位数，再对 70 批取中位数，为 57.8%。留存一张私有 TRAIN 局部叠图，目视核对 mask 与 P3 网格。以上曝光含跨轮重复。

每批最少 104 个无向正对和 582 个无向负对。正对中位数 336，有效 anchor 最少 37、中位数 56，缺少有效 anchor 的批次为零。

| 类别 | 增强后目标曝光 | 有效 anchor 曝光 | 有该类 anchor 的批次 |
| --- | ---: | ---: | ---: |
| Burn | 942 | 942 | 70/70 |
| Crack | 309 | 305 | 64/70 |
| Dent | 415 | 414 | 69/70 |
| Material missing | 590 | 590 | 70/70 |
| Tears | 143 | 124 | 42/70 |
| Tip curl | 92 | 59 | 21/70 |
| corrosion | 1,602 | 1,601 | 69/70 |

独立 CPU FP32 前向和两路梯度复查使用与 GPU 完全同一 SHA256 的第 1 批原生增强 TRAIN 32 图。只观察 BN 和梯度，接受更新为零。原始 TRAIN 有 632 个来源组，各类原始来源组数依类别顺序为 139／75／112／150／43／18／193。批内有有效 anchor 时，各类的来源组数中位数为 7／4／5／7／3／2／9。

在 TRAIN 固定批次 1、11、21 上分别求主任务与独立 SupCon 梯度，未调用优化器。梯度下游可达边界以网络层号和每层非零梯度张量数记录。P3 同图前向只更新一次原生 BN。断开辅助分支的原生对照在同一首批检查 loss、梯度和 Buffer 位级一致。三个梯度 cosine 为 −0.0858、−0.0777、+0.0625。主任务用原生 batch-sum，SupCon 用类均值。预检按 `min(1, 0.05 / max(||g_S||/||g_Y||))` 选 Smoke 权重，得到 `λs=1`。此时三批的相对梯度范数为 0.214%、0.241%、0.344%。这是有意保守的工程权重。正式协议需审查辅助强度，尤其是小比例是否足以形成科研上可辨识的表征干预。预检只量到短时初始化附近的信号。

## 有界 J0／J1 Smoke

两臂同一冻结权重、TRAIN 668 图／1,266 实例、seed42、640、batch32、nbs32、每轮 21 批并保留 28 张尾批。沿用 E0 的 AdamW、FP32 主参数和 CUDA AMP，调度按 100 轮的相同公式计算，执行在 70 次尝试处截断。Warmup 长度 63，之后覆盖 7 次实际更新。每次真正进入优化器前先在同一实验根目录的独占预算中记录一张持久尝试凭证，累计每臂上限 120。首步与末步均保留工程权重并严格重载模型、投影头、EMA、AdamW、GradScaler。原生预测验证只用合成张量。

| 实测 | J0 | J1 |
| --- | ---: | ---: |
| 有效更新／原生 AdamW 调用 | 70／70 | 70／70 |
| Warmup 后有效更新／AMP 跳步 | 7／0 | 7／0 |
| 原图／增强前实例／增强后实例曝光 | 2,228／4,232／4,093 | 2,228／4,232／4,093 |
| 优化器唯一参数张量数 | 291 | 295 |
| 峰值 CUDA allocated／reserved | 4.66／5.02 GiB | 4.69／4.98 GiB |
| Warmup 后中位 forward＋backward | 173.1 ms | 174.7 ms |
| Warmup 后中位含数值审计更新 | 215.8 ms | 218.4 ms |
| 从运行目录创建到成功回执落盘 | 49.59 s | 51.41 s |

70 对完整增强张量 SHA256、原图顺序及原生优化器分组和 LR 匹配。J0 也与历史 Exp14 E0 的初始原生状态和前 70 批账本匹配，70 批原生总 Loss 与分项逐值相同。该对照覆盖了当前运行的 70 批。J1 的 SupCon Loss 在 3.568–4.501。三个独立辅助梯度探针均到达 backbone 0–10 的 120 个权重和 BN affine 张量、neck 13／16 的各 12 个张量及投影头 4 个张量。Segment Head 及 P3 下游只收主任务梯度。联合更新实际改变 backbone 和投影权重，原生 BN 在前向中进位，辅助计算未更改其 Buffer。

显存 allocated 峰值增加约 38 MiB。计时在同一张 AutoDL RTX 3080 Ti 12 GiB 上完成。主循环计时从 CPU 数据审计结束后开始，整臂时间另含源数据校验、区域检查和 checkpoint 抽查。当前开销支持将 640／batch32 带进正式审查，70 步计时用于短程工程判断。原始 TRAIN、专业标签和 split 通过前后哈希复核。VAL 和 TEST 模型推理均为零。

**建议 `READY_FOR_J0_J1_FORMAL_REVIEW`。** 未来正式比较以匹配的 J0/J1 配对重训为准，冻结共同主任务、期末 FP32 checkpoint 和 VAL 口径。先以 TRAIN 梯度证据审查一个有科学效力的固定辅助强度，特别是当前 `λs=1` 在初始化附近只占 0.21%–0.34% 主任务骨干梯度。再由独立 Gate 决定两臂是否进入正式训练。效果、稳定性及额外种子留给各自的授权。

## J2 只读预审

A9 的 B 数据可复用 545 个 Polygon 候选、376 个 TRAIN 来源、历史人工与原图/标注哈希、版本绑定、Aligned、Valid Mask 和变换矩阵。A9-D0 的数值窗口试验有 541 个达到试验像素支持阈值。36 个目视试例中有 6 个具备 G1 研究可行性并待独立复核，15 个缺独立边界证据、11 个有视觉歧义、4 个因叠层或纹理差异失败。历史 180 个主建议仍需确定最终像素资产。正式可信 Pair 仍为零。[B 候选](../exp14/EXP14_A9_FINAL_DATASET_CONSOLIDATION_HANDOFF.md) · [G1 试验](../exp14/EXP14_A9_D0_ADMISSION_ROOT_CAUSE_REVIEW.md)

下一道数据 Gate 可从已留证的六个试例开始，逐一绑定 G0 的来源、Polygon 和精确修复版本，再核验 G1 的同一机械部位、完整目标与修复覆盖、可比局部结构及独立机械边界点。保留 UI、纹理或残留缺陷捷径审查，区分用于拟合与用于验收的证据。区域级 J2 可按 G1 设计，精细像素损失另需 G2 证据。A10 的整图空标签准入不替代这些证据。本轮 J2 训练为零。

## 复查入口

在独立 AutoDL 运行根目录，用固定 COCO checkpoint、已冻结运行部署和空的私有预算目录执行以下入口。设置独立的 `YOLO_CONFIG_DIR`，并保留原生运行日志。Smoke 两臂共用预算目录和相同代码的预检回执。CPU 真实批重放以同一预检回执调用 `tools.exp15.verify_cpu_train`。失败轮次继续保存在各自目录。

```bash
python -m unittest -v tools.exp15.test_ja
python -m tools.exp15.ja preflight --release "$release" --weight "$weight" --budget "$budget" --output "$preflight"
python -m tools.exp15.ja smoke --release "$release" --weight "$weight" --budget "$budget" --preflight "$preflight/summary.json" --arm j0 --output "$j0"
python -m tools.exp15.ja smoke --release "$release" --weight "$weight" --budget "$budget" --preflight "$preflight/summary.json" --arm j1 --output "$j1"
```

私有审计归档 SHA256 为 `92ec4234c732df48a7d6ee88f2196abd6f17b2ee424fc3a6ba169c69d34e42c1`。失败证据保留在独立试次。初次 CPU 合成测试暴露了负例坐标未越过栅格容差以及比较容器类型的问题。后续零更新预检分别暴露了 native dataloader 路径含 `..`、对照模型需配置原生 loss 超参数，以及 loss 分项为字典。均已修复并重新验证，对应失败试次接受的优化器更新均为零。

[脱敏结果](../../results/exp15_ja/summary.json) · [运行入口](../../tools/exp15/ja.py) · [区域损失](../../tools/exp15/regions.py) · [脱敏汇总入口](../../tools/exp15/summarize.py)
