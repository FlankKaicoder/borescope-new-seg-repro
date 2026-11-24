# R3-D：共享前段内的两次Mean局部对照建议

NOT_AUTHORIZED / NOT_EXECUTED。R3-C两次额度已经耗尽，本文件不增加任何预算。

R3-C显示：在冻结32张TRAIN/62GT上，只纠正共享0–16的48个BN Mean即可恢复固定conf=.25的TP19/FP6，后段单独纠正仍TP0/FP2599。为区分前段内部的功能补偿范围，建议两个互补条件：

| 调用 | 使用当前路径batch Mean的范围 | BN数 | 其余Mean / 全部Variance |
|---|---|---:|---|
| 1 | `model.0`至`model.10`及全部子模块（Backbone） | 40 | 原running / 原running |
| 2 | `model.11`至`model.16`及全部子模块（P3 Neck） | 8 | 原running / 原running |

两组后段17–23均不干预。BN名称从冻结模型精确解析并绑定；不能用字符串包含方式误选模块。前段存在skip/concat连接，此划分不等于两个彼此独立的线性网络。

复用原Last MASTER文件SHA `063e473cf0b878f105c85790be47332a30d8192e09a14ad28ab2180b85ba6e2d`、packet SHA `802114bde743ce7c58e997be279987148f33d91898d43b511ad5040939edd929`及原输入、TRAIN allowlist。封存Running、全网Mean和R3-C前段Mean作为只读参照，不重跑。

上限恰好2次新模型前向、0次构造器/合成前向，独立账本、先attempt后commit、禁止自动重试。FP32、AMP/TF32关闭、CUBLAS确定性、独立副本；记录所有90BN真实输入/输出矩及两阈值NMS描述结果。CPU范围/预算/状态保护测试后才可执行。资源上限10GiB/15分钟。

原权重、buffer、Protocol、Runtime、授权、Checkpoint及所有历史账本只读；不训练、不backward、不优化、不校准、不保存模型、不访问VAL/TEST。不调整conf、NMS、类别映射或匹配规则，不以追求TP19决定追加调用。

判读预案：只有Backbone纠正恢复，支持该段纠正的充分性；只有P3 Neck恢复，说明下游局部补偿也可能足够，不能反推Backbone健康。两者均恢复，提示多条充分补偿路径；两者均不足而联合前段可恢复，提示组合干预作用，不能自动推断每层必要性。所有情况同时看FP、几何与层间统计，不只看TP或检测数。

后续仍需区分统计跟踪、权重敏感性与训练动力学的上游成因。该两次局部试验不检验Pair因果、跨batch泛化或正式Mask AP；完成后停止，不自动扩大范围。
