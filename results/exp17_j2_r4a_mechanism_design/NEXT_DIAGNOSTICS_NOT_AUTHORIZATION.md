# R4-B 多批次统计与 R4-C 导数机制：拟议范围

`NOT_AUTHORIZATION / NO_NEW_EXECUTION`。此文件给出可审查范围，不签发 Gate PASS。R4-A 资产与数学检查完成，R4-B 执行器未准备，R4-C 图等价性尚未测试。

## 共用保护

- 保留原 Protocol、Runtime、COCO、Manifest、正式训练/VAL授权、执行 seal、永久 claim、80次 pass、模型和日志原字节。新研究使用独立诊断目录、独立不可覆盖调用 ledger；不调用正式 executor、NativeAdapter 构造器、validator 或 recovery。
- 只访问冻结 TRAIN packet。R4-B 无 N/Valid 解码；R4-C 仅按原 Pair index、ReplayTrace、N/Valid/Matrix SHA 准备已准入 A/B/N。不得访问 VAL/TEST，也不调整评价阈值、科学参数、样本选择或模型权重。
- 克隆模型，保护 tensor、模式、BN、RNG；诊断中临时统计/梯度不回写原 Checkpoint，不保存校准模型。失败保留证据，不自动重试或换批次补额。
- 使用当前绑定环境和严格 FP32、TF32关闭、确定性/CUBLAS设置。资源上限为本次单卡进程 allocated 10 GiB；R4-B总15分钟、R4-C总20分钟上限。构建具体执行器时须实现超限停止；OOM不自动降 batch/分辨率。无梯度 R3 的显存占用不能保证 R4-C 能容纳计算图。

## R4-B：6次新的 A 前向，0次反传

问题：同一冻结权重的 running 统计失配，是否在事先冻结的其他批次上复现？

模型：Epoch100 Last MASTER，native文件 SHA `063e473cf0b878f105c85790be47332a30d8192e09a14ad28ab2180b85ba6e2d`，native tensor SHA `d222a70d7570680459359a36a682312690b052cc10197f4916ec1df66492782f`。直接读取既有原生序列化模型，严格核对模块非持久属性，不调用会产生 stride dummy forward 的构造器。

| Packet | 图数 | running | functional batch-both |
|---|---:|---|---|
| epoch000/position1 | 32 | 复用封存 R3-B | 复用封存 R3-B |
| epoch000/position7 | 32 | 新调用1 | 新调用2 |
| epoch000/position11 | 32 | 新调用3 | 新调用4 |
| epoch000/position21 | 28 | 新调用5 | 新调用6 |

四个 packet 的准确 SHA/张量身份见 [asset_bindings.json](asset_bindings.json)。新调用涉及92张来源、168个增强后实例，每种模式各一次，共184次图像曝光；合并历史后为124张/230个实例。尾批28张保持原样。无新抽样/增强，无按结果选择的替换。

两模式均保持检测/分割 head 为 eval、模型原参数不变。running 使用原统计；batch-both 用无持久缓冲写入的 functional BN，在同次前向中取当前 batch Mean/biased Variance。后者是诊断反事实，不能宣称是正式训练路径的完全复现。记录90个 BN 的输入/输出逐通道 moments、归一化偏差、原始分类/框输出与原有固定阈值 Box 统计；不计算/宣传新的正式 mAP。

重点观察同一模式跨batch的均值/方差分布，以及模型功能变化。区分同一模式同一层的输入来源；不同上游 BN 模式会改变下游输入，不能把它们当同一分布合并。跨批汇总按实际 B×H×W 计数加权，方差包含批次均值间差异；不直接平均4个方差。保留逐批结果，不以平均值掩盖反例。

原先 `r2_inputs.load_packet` 固定32槽/位置1，不能直接用于28张尾批。准备时仅在独立诊断工具中按已绑定四个 spec 泛化校验；针对32/28、错SHA、错来源/位置、重复调用做 CPU 测试。静态计数/桩测试不允许偷偷做模型 dummy forward。保护非持久属性与 IsolationViolation，不绕过失败。

验收：6个唯一 claim/完成回执；每个实际前向计数1；0 N、0 autograd、0 synthetic、0optimizer、0VAL/TEST。历史位置1两结果只读关联，不重新 claim。新结果及 moments 有完整SHA；原资产和模型状态不变。达到范围即停止，不自动转 R4-C。

## R4-C：后续单批次导数对照，单独范围

只提出一个终点、一个固定batch的诊断。完整2100步 MASTER＋projection，packet位置1；原 A/B/N资格、SupCon8、Pair4、margin.20、temperature.10不变。恢复原保存 `args.yaml` 并严格重绑 criterion；禁止用原生导出的 `args=None` 或环境默认损失参数。

两张图：①原 detached replay_a；②只改变 A统计导数路径的 connected-statistics 反事实。它不是正式协议允许的新训练模式，不能用于继续训练。两个图的 A/N前向值、P3、区域embedding、组件loss必须匹配，输入/资格必须一致。

直接给 `F.batch_norm` 的 running_mean/var 传入 requires_grad 张量可能不受支持，且原 `auxiliary_bn` 明确拒绝非detach统计。不得修改/绕过它。可在独立私有诊断 helper 中研究零值补偿路径：保留原 N 输出 `y_base`，增加 `v - stopgrad(v)`，其中 v 用 graph-connected A统计、detached N输入和 detached affine 参数计算。其数值增量为0，导数仅增加 A统计路径；必须先做小型 CPU 数学/图梯度验证，再验证真实图前向相等。该构造是明示的 surrogate derivative 干预，不得冒充原始执行。

| 操作 | 两图合计上限 |
|---|---:|
| A全网特征遍历 | 2 |
| N的0–16前缀特征遍历 | 2 |
| YOLO、SupCon8、Pair4 各一次 autograd.grad | 6 |
| 每类额外梯度、联合额外反传、checkpoint重计算 | 0 |
| 构造器stride forward、optimizer、VAL/TEST | 0 |

准备阶段另设最多12次 CPU 小张量 `autograd.grad`，用于验证补偿项的导数与独立数值差分；不构造神经网络、不读取图像。它们不冒充上述6次真实模型组件梯度；R4-C拟议 autograd 总上限因此为18次（CPU fixture≤12＋模型组件6）。本轮 R4-A 仍全部为0。CPU验收失败即停止，不进入真实模型调用。

一次 `model(batch)` 可能有 dict→loss→tensor 的内层分派，应按实际特征遍历计数并另外披露入口调用；N是手动prefix，不能遗漏。不得直接调用会额外做每类probe或两种BN路径的完整旧诊断函数。

每次组件梯度请求可同时收集参数和Neck16四个BN的 A/N 输入梯度，不增加额外autograd调用。统计共同平移/尺度方向分量、Pair4相对YOLO+SupCon8的方向/范数和层级参数分量。共同平移是虚拟输入方向，Conv无bias；只有实际参数分量能进一步讨论可实现更新。梯度本身与 AdamW 历史 moments 分开解释；不调用optimizer.step或用公式假装完成真实更新。

成功只支持该终点/批次的机制参与。若Pair资格为空、hinge不活跃、两图前向不相等、显存不足或任何绑定/状态冲突，保留结果并停止；不替换样本或加做历史epoch。是否需要早期epoch或独立训练对照，依证据另行提出，不能从本预算自动扩展。

## 解释边界

- 跨批次恢复支持当前统计失配的普遍性，不证明历史单步滞后。BN momentum=.03的理论窗口不是原因证明。
- 实际 detach 梯度差异支持一个参与机制，不证明全部历史 drift 由 Pair 造成；没有历史逐通道梯度可供补录。
- 原始 Last/Best/EMA 与正式 AP 不变。任何后续修复、BN校准、重新训练或正式评价均不包含在上述范围内。
