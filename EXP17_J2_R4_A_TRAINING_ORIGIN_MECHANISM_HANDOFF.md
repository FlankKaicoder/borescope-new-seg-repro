# Exp17-J2-R4-A：训练诱因证据与机制设计

日期：2026-11-24（Asia/Shanghai）。依据提交 `d11b9c7b6d20fa3bf94bc44aa3fba6e5284b9c72` 及封存 R2/R3 证据。

状态：`EXP17_J2_R4_A_MECHANISM_DESIGN_COMPLETE_PENDING_CHATGPT_GATE`。科学状态仍为 `HOLD_TRAINING_ORIGIN_UNRESOLVED`。

本轮完成只读资产核对、梯度机制的合成数学验证和后续预算设计。新增模型构造/前向/反传、GPU、优化器更新、VAL、TEST 均为 **0**；没有修改正式代码、模型、Protocol、Runtime、授权或账本。下一阶段数值实验尚未执行。本轮不是新的训练结果或训练修复。

## 1. 现在知道什么

原 Seed42 已完成100 Epoch、2100次有效更新、80次正式 VAL pass。Last MASTER Mask AP50–95 仍为0；J1 Seed42 为0.2727438798549706，差值 −27.27438798549706 pp。诊断 TRAIN 的 Box TP 不能替代正式 Mask AP。

R2 的模型/导出/融合对照未支持简单的导出错误。R3 的同权重、同输入统计干预表明，BN 统计来源会改变检测功能：全局 Mean-only 为19 TP/6 FP；保持 Backbone Mean 干预、再纠正 Neck16 的4个 BN，得到17 TP/7 FP；同条件纠正 Neck13 则仍为0 TP。上述均限于同一32张 TRAIN、62个增强后 GT、固定 conf=.25；不是新的 VAL 结果，也未证明 Neck16 单独充分。

因此，**推理失效的近端环节已定位到条件性的 BN Mean 失配；训练为什么产生该失配仍未确定**。继续逐层扫描的价值已下降，下一步应区分批次特异性、统计跟踪与 detach 梯度机制。

## 2. 本轮核对到的可用资产

| 原固定位置（零基 epoch=0） | 图像 | 增强后实例 | 用途 |
|---|---:|---:|---|
| 1 | 32 | 62 | 已有 R3-B running / batch-both 结果，优先复用 |
| 7 | 32 | 65 | 预先指定的新批次 |
| 11 | 32 | 56 | 预先指定的新批次 |
| 21 | 28 | 47 | 保留原尾批，不填充、不拆分 |
| 合计 | 124 | 230 | 124张互异 TRAIN 来源；不是随机独立总体样本 |

四份 packet 的文件 SHA、六张量 digest、归一化图像 SHA、ReplayTrace 和 TRAIN668 成员资格均通过。这里只检查封存张量/元数据，未解码 N/Valid 图像，未重新生成增强，也未重新判定 Pair 准入。新批次没有 Tip curl 实例；不能据此推广七类结论。

2100步完整 Checkpoint 文件 SHA 为 `79b237241517e9112687f9ecd155d21c096ce54243e9a8267b21a963d58c675e`。其 native MASTER SHA 与最终导出一致，保留4个 projection 张量（64→128→128）和294个参数的 AdamW 状态。最终原生导出本身没有 projection，且 `args=None`，不能直接用于 YOLO/SupCon/Pair 损失诊断。

已找到两次正式初始化保存的完整 `args.yaml`；逐行差异只有 `project` 与 `save_dir`。未来组件梯度诊断应使用 R1 初始化保存的参数，绑定其 SHA `d1dfef12c7d5808e8530443f84577eee40d09f42d01cffc47214efa2a42c495a`，再核对冻结 Protocol；不能借用当前环境默认值。损失参数包括 box=7.5、cls=.5、dfl=1.5、overlap_mask=true、mask_ratio=4。原 AMP=true；拟议机制探针明确采用 FP32，是新的诊断口径。

身份必须区分：完整 Checkpoint 内置 `checkpoint.state_hash` 是 `609f1ae560829ca9c5bad715a68f1022a36be85b7ce45d003a0464dc21e1ea85`；R3-A 的原始张量串接指纹是 `fead9b162035fcca97c3d129e47494e12fbdf7e71df2af4409e4a22a0ca448da`；native `hash_state` 是 `d222a70d7570680459359a36a682312690b052cc10197f4916ec1df66492782f`。算法/覆盖对象不同，本轮分别重算通过，不把不同 digest 当作文件变化。

## 3. 新增数学证据及其限制

现有 `replay_a` 从 A 捕获 detached Mean/biased Variance，N 使用这些统计，但保留共享参数梯度；它不会额外更新 running buffers。80个历史 probe 只存分层汇总，无法恢复历史逐通道梯度。

对一个 BN 通道，令 A 的均值/方差为 μ、v，`s=γ/sqrt(v+eps)`，N 输出的上游梯度为 hN。A、N 同时加一个相同常数、并重新计算 A 统计时，归一化前向值不变。保留 A 统计的导数时，这一共同平移方向的总导数为0；detach 后，缺少统计补偿项，方向导数成为 `s × sum(hN)`，可不为0。

这是**相同前向值、不同导数**的可检验机制，优于把 replay_a 与 independent-N BN 直接比较：后者同时改变统计值，无法单独识别 detach 的作用。

纯 NumPy float64 的6组算例通过独立中心差分：一般输入、不同 A/N 数量、N 上游梯度和为0、γ=0、大偏移及负 γ；最大逐坐标误差约 `5.85e-9`。共360次合成标量目标计算，模型前向与 autograd 为0。详见 [数学记录](results/exp17_j2_r4a_mechanism_design/synthetic_derivative_verification.json)。

**这不是实际 J2 梯度，也不是根因证明。** hN 在实际模型里可能抵消；相关 Conv 没有 bias，因此虚拟通道平移不必对应一个可独立实现的参数方向；共享参数、其他 BN、SiLU、AdamW 历史 moments 与 Clip10 都会影响实际更新。尚不能说 Pair 导致了 drift，更不能据此修改 replay_a。

## 4. 推荐顺序与可证伪判据

1. **R4-B：先排除单批次补偿。** 冻结 Last MASTER，复用位置1的两份历史结果；位置7/11/21各做 original-running 与 functional batch-both，共6次新 A 前向，0 N、0反传。记录90个 BN 的实际输入统计与原 running 统计的偏差，观察恢复能否跨预定批次出现。它可以支持跨批次失配，不能还原过去的单步统计滞后。
2. **R4-C：再检验 detach 的实际导数作用。** 单个固定批次上构造前向完全相同的 detached/connected-statistics 两个诊断图；预算2次 A全网＋2次 N前缀、6次模型组件 `autograd.grad`；准备时另设最多12次CPU小张量导数验收，合计最多18次autograd。比较实际 Pair4 的 Neck16 通道方向与参数方向。此为独立后续范围；本轮只设计，尚未准备执行器或完成真实图等价性验收。
3. **最后结合既有历史轨迹解释。** 将新观测与原100 Epoch/2100步、80 probe、BN/Conv 轨迹、AdamW moments 对照。历史相邻更新的真实激活没有保存，不能事后补成现场证据。单个终点梯度无法分解整个训练的历史因果贡献。

若 R4-B 的新批次一致出现 running 失效、batch-both 恢复，说明此前现象不局限于一个批次；若只在位置1成立，应优先调查构成/增强敏感性，不能挑选新批次追加。若 R4-C 在前向相等前提下出现可观、可解释的梯度差异，支持该梯度机制实际参与，但仍不足以判定它是训练全过程的主要原因。未出现差异也只能削弱这个终点/批次上的解释。

完整操作边界、预算和验收见 [R4-B/R4-C设计](results/exp17_j2_r4a_mechanism_design/NEXT_DIAGNOSTICS_NOT_AUTHORIZATION.md)。

## 5. 裁剪、工程结论与交付

2100次 Clip10 触发仍作为真实训练现象保留。既有 probe 表明去掉 Pair 的代数参考梯度也超过10；不能把全触发归因于 Pair。Pair 的未加权比例解释时乘4，N_P3 指标已含4，不再重复乘。clip 系数不是 AdamW 参数变化比例；本轮不增加探针、不改 Clip10。

若发现实际运行与冻结实现不一致，才归为工程缺陷。若实现符合协议，但统计/导数机制导致训练与推理不配套，应归为这套 J2 配置的训练动力学失败。当前不能从一个 Seed42 的全零结果推断所有 Pair Ranking 方法无效，也不能将诊断恢复解释为正式实验已修复。

本轮核对7个关键文件及4个固定 packet 前后 SHA 不变；CUDA 未初始化。两次早期只读检查被依赖导入的临时目录写请求拦截，未进入模型执行；后改为按原源码 AST 提取3个纯哈希函数，成功完成只读检查。失败日志与源码版本保留。正式源码未改，新增文档及汇总显式交付，旧 G4 未提交改动保留。

后续停止点由新的数值范围决定：R3-E两次调用已经耗尽，当前批准的是训练诱因设计；本报告没有创建新的执行授权。原 Last=0、80次正式 VAL、Protocol/Runtime/Checkpoint 和科学边界均保持不变。
