# R5 Epoch5 HOLD 后的最小研究决策提案

当前状态为 `EXP17_J2_REPAIRED_SEED42_EPOCH5_SCIENTIFIC_HOLD`。

R5已完成105次有效更新和Epoch5全部事件。MASTER及EMA的标准Box/Mask AP、Precision、Recall均为0。两个fixed结果均有46200个检测、0TP、46200FP、296FN。四个预定TRAIN A输入统计位置的D约为0.16–0.68，全部低于预定门限3。Journal共221行，105次更新与5个AUX全部提交，pending为空。完整105步Last、periodic、两角色Best和四个pass回执已保存。

用户批准的R5计划明确规定Epoch5全零AP时停止后续训练。本次因此保留105步状态，尚未执行Update106。科研参数、原Protocol、原J2结果及原正式VAL预算保持冻结。工程记录错误已自主修复并完成58项CPU测试和真实AMP Smoke。

## 已完成的离线判断

- 回执记录了两个不同的native模型状态，四份原始评价SHA、角色、置信度、输出和模型摘要闭合。原始评价未保存逐图框或logits。
- J1逐步日志SHA为 `d48afd642240ceb2e58306883d39bd0aa71385cddf4353dc2cc5497be832da72`，与已发表交接相符。J1前五轮YOLO Loss均值依次为439.453748、467.981444、469.672872、463.512971、452.628501，105步也全部触发Clip10，AMP skip0。R5的相应均值为440.679044、470.688516、471.459630、475.541688、461.984735。早期Loss上升和持续裁剪本身不足以解释R5的特有失败。
- J1保存了105步Checkpoint，SHA为 `09c9b216ba24c73c14629057dae4078fc029c9404ed5f9be83db880604cbb1ea`。现有J1正式评价只覆盖Epoch100，缺少Epoch5评价结果。
- R5四个固定probe中，Pair4与主联合梯度范数比约0.97%–3.34%，Pair105步均参与。Neck16 cv1/cv2的Epoch5权重范数比原J2同期低约10.59%和17.74%。现有四层TRAIN A偏移没有原J2同口径历史激活可直接对照，也不能代表完整running-eval路径。

当前可以确认早期功能Gate失败。R5的Epoch100 Last结果仍然缺失，无法形成修复后的百轮效果或追加种子的结论。

## 推荐追加一个独立、最多三次的TRAIN只读对照

这项新范围需要审查后启用，当前尚未执行。它直接影响下一步选择继续相同训练预算，还是先修复剩余的运行统计或模型功能问题。

复用原冻结TRAIN position1的32张图、62个增强后实例，以及受SHA保护的原封存CPU包。比较以下三个预先固定的状态，每个仅一次完整模型推理。

| 次数 | 模型状态 | 归一化 |
|---|---|---|
| 1 | R5 Epoch5 MASTER | 原保存Running Mean/Variance，正常eval |
| 2 | 同一R5 Epoch5 MASTER | 仅在独立副本上使用functional当前batch Mean及biased Variance；不写入buffers |
| 3 | 已保存J1 Seed42 Epoch5 MASTER | 原保存Running Mean/Variance，正常eval |

三次使用同一输入、七类、FP32、原epsilon及affine顺序、NMS和匹配口径。由每次唯一前向的输出按原`.001/.25`两种阈值后处理，记录Box/Mask TP、FP、FN、检测数量、几何匹配及必要的输出摘要。阈值与max_det300保持原口径。构造器stride探测单独计数，初始化和严格重载先做元数据核验。参数、buffers、training flags和RNG前后均核对。

预算为最多3次完整TRAIN模型前向，0N分支、0Autograd、0Optimizer Update、0VAL、0TEST。使用独立诊断代码、目录、永久调用账本和输入/模型/源码身份记录，限单卡10GiB和10分钟。原105步Checkpoint及四次VAL记录继续封存。该预算独立于已结束的4A+4N Smoke和正式VAL预算。

## 结果如何改变决策

- R5 Running失效、当前batch统计恢复检测时，优先审查四个sentinel之外及eval传播的归一化问题，连接统计梯度本身不足以保证推理统计匹配。
- J1同期Running有正确检测而R5两种统计均无正确检测时，优先检查早期权重功能和优化行为。
- J1同期也无正确检测时，获得了同阶段TRAIN功能参照，仍需保留TRAIN与VAL的口径差异。此时可提交从原105步连续观察至Epoch10的Gate修订，继续使用原2100/80总预算，无需重做前105步。
- R5在同一TRAIN输入上Running已有正确检测、冻结VAL仍全零时，优先审查数据域、预处理和评价路径差异。

完成该有明确决策用途的对照后提交结果。任何解除Epoch5科学HOLD的继续训练，都需要明确修订已冻结的早期停止决策，并绑定原105步状态与原预算。
