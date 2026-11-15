# Exp15 J-E0 多随机种子验证预注册与交接

本轮以已审计 HEAD `bca1d48f2a3004fe843db4a259f52e3ee360d0ff` 和独立 Result Gate `EXP15_JD_PASS_SINGLE_SEED_POSITIVE` 为输入。J-E0 的授权范围为协议准备、工程适配和 TRAIN-only 预检。正式训练和冻结 VAL 模型推理各需下一阶段的独立 Gate。

## 验证问题与完成定义

检验共享 Backbone 的 Polygon SupCon 是否在相同正式协议的不同随机种子中改善七类 YOLO11n-seg 的实例分割。固定种子集合为 42、142、242，每个种子包含 J0 和 J1。seed42 是发现结果，直接使用已审计的 J-C 训练终点和 J-D 实测 VAL。seed142 和 seed242 各完整执行两个新的正式实验臂，再逐对接受独立 VAL 审计。完整结果集合由三个有效的种子配对组成。

| 种子 | J0 | J1 | J1−J0 Mask mAP50–95 |
| --- | --- | --- | ---: |
| 42 | 复用 Exp15 J-C/J-D，0.249512 | 复用 Exp15 J-C/J-D，0.272744 | +0.023231 |
| 142 | 待独立授权训练及评价 | 待独立授权训练及评价 | — |
| 242 | 待独立授权训练及评价 | 待独立授权训练及评价 | — |

正式执行次序为 seed142 J0 → J1，再执行 seed242 J0 → J1。每臂独立验收后再启动下一臂，使用对应 seed 和实验臂的独立运行目录、优化尝试账本、日志和 checkpoint。既有 seed42 训练结果保持只读，Exp10 checkpoint 在本协议的训练输入清单之外。种子和实验臂集合已冻结，所有纳入条件与科学解释规则在观察新增种子的 VAL 前固定。 [J-C 训练交接](EXP15_JC_HANDOFF.md) · [J-D 实测摘要](../../results/exp15_jd_val_seed42/summary.json)

## 共同训练口径

逐项沿用 J-B 正式协议。此阶段唯一实验因素为 J1 在共享 Backbone 上加入现有 Polygon SupCon。J0 使用原生 YOLO11n-seg 损失，J1 使用相同原生损失加 `8 × SupCon`，温度固定 0.10，保留现有 ROI、mask raster、来源组和 anchor 规则。

- 从冻结 COCO 权重开始七类原生模型初始化。COCO 文件 SHA256 为 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`。同 seed 两臂原生状态和 BN 状态必须逐字节同哈希，J1 投影头单独核验。不同 seed 可以形成不同的初始化和采样轨迹。
- 两臂只使用冻结的原始 TRAIN 668 张图、1,266 个实例。冻结划分 SHA256 为 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`。使用原有 `epoch_seed(seed, epoch)` 和 worker 派生逻辑，每轮固定 seed 下的无放回随机采样。同 seed 逐批核对图像有序身份、增强张量哈希、原始及增强后实例计数。
- 两臂 640 输入、batch 32、梯度累积 1、21 批每轮、最后一批 28 张、100 轮、2,100 次有效更新。每臂原图曝光 66,800 张次，增强前实例曝光 126,600 次。AdamW、原生线性 LR、63 个 warmup 位置、FP32 master、原 CUDA AMP／GradScaler 和全局 L2 裁剪阈值 10 都沿用原协议。两臂逐批核对原生参数分组、LR 和 scaler 轨迹。
- 每 5 轮保存独立 checkpoint，正式评价只选第 100 轮 FP32 master。检查更新步数、模型和优化器身份以及严格重载。J1 评价使用去除投影头后的原生七类 FP32 模型。EMA 单独保留，正式评价输入清单仅含两臂的 epoch100 FP32 master。
- 共享的主任务配置、初始化绑定、数据快照及环境实际版本应进入机器协议和脱敏证据。seed42 的协议、源文件指纹和历史 checkpoint 使用原有审计记录。 [J-B 正式协议](../../results/exp15_jb/formal_protocol.json) · [J-E0 机器协议](../../results/exp15_je0_multiseed/formal_protocol.json)

## 冻结 VAL 与统计计划

完整训练和同 seed 配对审计通过，且获得独立 VAL Gate 后，对每个新增 seed 复用 J-D 的评价器、冻结文件清单和类别映射。每臂在相同的 154 张冻结 VAL 图像和 296 个目标上执行标准与固定点评价各一次。使用 Ultralytics 8.4.117 原生 validator、640 输入、batch 32、CUDA FP32、TF32 关闭、NMS IoU 0.70、max_det 300。标准置信度 0.001，固定置信度 0.25，固定点使用同类 Mask／Box IoU ≥ 0.50 的原生一对一匹配。尺寸沿用文件中冻结的 TRAIN 相对面积分位数。

主要终点是 Mask mAP50–95。对每个 seed 报告原生未舍入的 `J0_s`、`J1_s` 和 `Δ_s = J1_s − J0_s`，计算 `mean(Δ)` 和样本标准差 `sqrt(sum((Δ_s − mean(Δ))²) / 2)`。另外报告各臂三种子的均值、样本标准差及正收益计数。所有统计先用底层精度计算，最终显示值可舍入。以 `Δ_s > 0` 计作正向，零差值计入非正向。三种子结论按下表判读。

| 正收益种子数 | 预注册解释 |
| ---: | --- |
| 3/3 | 较强的初步跨种子稳健性证据。并列呈现效应幅度和类别代价。 |
| 2/3 | 混合证据。结合均值、离散度及错误分解审查。 |
| 1/3 或 0/3 | 稳定增益未获支持。保留各 seed 的完整结果。 |

只有三对均验收通过才发布三种子汇总。任一配对尚未完成或无效，保留已有结果和异常记录，多种子结论待审。新增两对的方向单独列出，以保留发现结果和复验结果的来源。统计为此冻结数据划分上的描述性证据。

同步报告原生 Mask Recall、Precision、逐类别 AP，以及固定置信度下的 Box／Mask TP、FP、FN 和四档 Mask Recall。各类别和尺寸以配对差值解释，GT 为 0 的 Recall 保留空值。预先关注 Tears 的 Mask AP50–95、Dent 的 Tiny Mask 召回、Crack 的固定点 FP，以及 Burn 和 corrosion 的 Mask AP50–95 与固定点 FP/FN。seed42 的参考差值包括 Tears Mask AP50–95 +8.666 个百分点，Dent Tiny 命中 11/26 → 17/26，Crack Mask FP 2 → 12，Burn Mask AP50–95 −0.077 个百分点，以及 corrosion 召回和 FP 代价的方向分歧。后续保留全部逐类结果。 [J-D 整体与逐类表](EXP15_JD_HANDOFF.md) · [J-D 逐类尺寸](../../results/exp15_jd_val_seed42/size_recall.csv)

## 验收边界与停止条件

- 预检只消费 TRAIN 和已存在的审计回执。对每个新增 seed 计划检查前 70 批增强输入的配对。通过零更新身份与配对检查后，可对每个新增 seed、每臂运行 1 次有界 GPU 优化器尝试，独立账本和输出目录的授权硬上限为每臂 5 次。预检结束提交独立正式训练 Gate。
- 身份核验覆盖代码、协议、COCO、冻结 TRAIN 图像、标签、split、组身份和运行时。发现身份变化、同 seed 原生初始化或输入轨迹不匹配、原生 optimizer 分组、步数或 LR 偏离协议时停止，保留目录和首个不一致处。
- 非有限数值、原生 AMP 跳步、无有效参数更新、SupCon anchor 路径失败、checkpoint 重载失败、资源不足或运行中断时，保留尝试账本、日志及已有状态，由独立 Gate 判定处置。科学协议改变须重新预注册和审计，种子、权重、温度、训练预算、数据划分、评价阈值及终点评价规则在本协议内固定。
- 下一阶段正式 TRAIN Gate 与独立 VAL Gate 分开。TEST、辅助 CE、J2、图像修复及论文仓库均在本实验范围之外。既有实验文件、权重、正式回执和结果保持原样。

## J-E0 工程预检回执

**`EXP15_JE0_MULTISEED_READY_FOR_FORMAL_GATE`。** 冻结机器协议 SHA256 为 `1e114398c58a87337a1addbc4392b4485e651f928e95ca9d8c189af52972a305`。执行适配提交为 `82b3239bfb589ed8716308b0fed35c529dffdcf9`。完整预检回执 SHA256 为 `ebe00b9c0c2a4150837110f737a1a7afeffe3886a391bca6ded59fcd184d1e02`，与服务器副本一致。正式授权将绑定这两个 SHA256。此次没有生成正式授权。

新适配层在原生训练器构造前同时向 Ultralytics 初始化和既有的 epoch／worker 随机种子入口注入 seed，只改变这项共同配置。复用 J-B 原有更新循环、J-A 原生模型初始化和投影头 RNG 隔离，以及 J-C 独立审计。历史 Exp14 和 J-A 至 J-D 文件指纹保持一致。本地 J-A／J-B／J-E0 合计 11 项 CPU 单元测试通过，AutoDL 的 4 项 J-E0 测试也通过。

| 新 seed | 原生初始状态 SHA256（同 seed 两臂一致） | 前 70 批匹配 | 第 100 轮回放匹配 | 前 70 批最少有效 anchor | J0／J1 有效 Smoke 更新 |
| --- | --- | ---: | ---: | ---: | ---: |
| 142 | `3b7779eb31537a8b402b500741d0187a7efcfce5c66b0ec5482c10e96d6514bc` | 70/70 | 21/21 | 36 | 1／1 |
| 242 | `de598e518c76707b3c0c30f15d28cfb912fadb3db1c58a6b767358c6b8a05b01` | 70/70 | 21/21 | 38 | 1／1 |

两种子各通过 91 批零更新回放，逐批匹配原图顺序、完整增强张量和实例计数。两对的同 seed BN、原生优化器分组和首步原生损失相同。复算前 70 批 warmup 和学习率的最大绝对误差为 1.39e−17。两种子首批和第 100 轮首批的图像顺序哈希各不相同，七类原生模型初始化哈希也不同。668 张原始 TRAIN 图对应 632 个来源组，预检前后复核 1,336 个图像与标签文件。不同种子的 COCO 来源权重相同。

四个独立 Smoke 账本各记录 1 次尝试，均为有效更新，AMP 跳步 0。每个 checkpoint 通过严格重载并绑定 seed、实验臂及协议。J1 四个投影参数张量均有非零梯度。单步更新发生在 warmup 内，首 70 批的零更新回放另覆盖 warmup 后的 LR 和数据轨迹。

实测 AutoDL 为单张 RTX 3080 Ti 12 GiB，PyTorch 2.8.0+cu128、CUDA 12.8、Ultralytics 8.4.117。Smoke 峰值保留显存约 5.01 GiB，结束时盘面剩余 6.79 GB。按四臂每臂 21 个 checkpoint 估算需 3.90 GB，正式开跑前逐臂复核剩余空间。私有审计归档 SHA256 为 `ccf67fd628d6deaf09c0ff62f3bc8985d7497e26c580758262950232e1daec45`，包含原始运行和预算账本。四臂权重保持在各自的私有 Smoke 输出目录。

正式 100 轮轨迹和跨种子结果仍待验证。现有 J-D 入口的权重身份锁定 seed42。后续 VAL Gate 需绑定新增种子的试验终点，沿用冻结的 J-D 指标实现和参数。此次正式训练、seed42 重训、VAL 及 TEST 推理均为零。

[脱敏预检摘要](../../results/exp15_je0_multiseed/summary.json) · [正式 Gate 绑定回执](../../results/exp15_je0_multiseed/preflight.json) · [训练适配](../../tools/exp15/je0.py) · [独立对账](../../tools/exp15/summarize_je0.py)
