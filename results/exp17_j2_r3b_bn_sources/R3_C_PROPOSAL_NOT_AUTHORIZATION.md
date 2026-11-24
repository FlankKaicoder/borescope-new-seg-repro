# R3-C：两次局部Mean来源对照建议

NOT_AUTHORIZED / NOT_EXECUTED。本文件只提出下一轮最小调查，不延长R3-B已耗尽的4次额度。

R3-B显示全网Mean-only恢复Box TP19，全网Variance-only仍TP0。下一问题是在哪一段纠正Mean足以恢复，而非立即归因Pair或调整科研参数。

建议两个互补条件，均在原J2 Last MASTER、原32图62GT TRAIN packet的独立副本上：

| 调用 | 使用当前路径batch Mean的BN范围 | 其余Mean | 所有Variance |
|---|---|---|---|
| 1 | model.0至model.16（含端点及全部子模块） | 原running | 原running |
| 2 | model.17至model.23（含Head及全部子模块） | 原running | 原running |

共享0–16包含Backbone与P3 Neck，不能将其称为纯Backbone。读取本轮已封存的Running/全网Mean-only作为同身份参照，新增预算只允许2次模型前向、0次构造器前向；不复用或重置R2/R3-B账本。CPU范围匹配和预算拒绝测试通过后再执行。

模型文件SHA `063e473cf0b878f105c85790be47332a30d8192e09a14ad28ab2180b85ba6e2d`；packet SHA `802114bde743ce7c58e997be279987148f33d91898d43b511ad5040939edd929`。FP32/TF32关闭、CUBLAS确定性、独立副本、逐层矩记录、既定两阈值及Box匹配均沿用R3-B。原权重、buffer、日志、授权及所有历史账本只读。无校准、backward、optimizer、VAL/TEST或模型保存。资源上限10GiB/15分钟。

若只有前段纠正恢复，说明前段纠正在该条件下足够；不能直接证明Pair致因。若只有后段恢复，可能是下游补偿，不能立即判Head原发缺陷。若两段各自都能恢复，说明存在多条功能补偿路径；若两段都不能恢复，说明跨段共同纠正可能必要，继续HOLD并另提范围，不自动增加调用。

这是局部充分性对照，不等于唯一致因定位、必要性检验或跨batch泛化。判读同时报告TP、FP、几何与激活，不能只看检测数量，也不能把TP19当成必须达到的优化目标。需后续明确批准并绑定脚本、输入和独立账本后方可执行。
