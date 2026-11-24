# R3-E：固定Backbone干预下的Neck13/16两次条件式对照

NOT_AUTHORIZED / NOT_EXECUTED。本建议不延长已耗尽的R3-D两次额度。

R3-D：仅Backbone Mean纠正仍TP0，Neck16偏差再次放大；仅Neck纠正为TP2/FP19；R3-C联合前段为TP19/FP6。下一步问题限定为：在Backbone Mean已纠正的相同条件下，纠正Neck13或16中的哪一组足以改善后续功能？不把TP19作为调参或追加调用目标。

| 新调用 | 固定干预 | 唯一改变的Neck范围 | 总纠正BN数 |
|---|---|---|---:|
| 1 | Backbone `model.0`–`model.10`的40BN Mean | 仅`model.13`全部4BN Mean | 44 |
| 2 | 同上 | 仅`model.16`全部4BN Mean | 44 |

两组其余Mean、全部Variance为原running。每个选中BN使用本次路径自身的batch Mean，不移植其他条件的激活。实际模块名称应从冻结模型解析，分别覆盖cv1、cv2及m.0.cv1、m.0.cv2；不按字符串模糊匹配。冻结Backbone条件是干预的一部分，不代表恢复训练或修改保存模型。

只读复用R3-D的Backbone-only和R3-C的共享前段联合条件，不重跑参照。Last MASTER SHA `063e473cf0b878f105c85790be47332a30d8192e09a14ad28ab2180b85ba6e2d`；32TRAIN/62GT packet SHA `802114bde743ce7c58e997be279987148f33d91898d43b511ad5040939edd929`。

预算恰好2次新前向、0次合成/构造器前向；独立副本和账本、先ATTEMPT后COMMIT、禁止自动重试。FP32、AMP/TF32关闭、严格确定性/CUBLAS配置、两阈值/NMS/七类Box匹配与此前一致。90BN全部逐通道输入/输出矩记录，保留原参数、buffer、模式及输入SHA。CPU精确范围/预算拒绝测试后执行。资源上限10GiB/15分钟。

没有backward、optimizer、校准、模型保存、训练、VAL/TEST、原始文件修改或新种子。新增诊断不能写入原80次VAL，也不能改变正式Last结果。

若某一组恢复，只说明该组在Backbone已纠正条件下足够；不等于原始故障唯一发生在该模块。若两组各自都改善，报告多条补偿路径；若均不足，报告组合依赖，不自动增加逐BN扫描。TP、FP、几何及下游均值/方差必须联合判读，不以检测数或分类logit均值单独判成功。

本建议止于模块级功能定位。无论结果如何，之后应单独审查“训练为何形成失配”的机制研究，而不是无限细分激活干预。当前没有授权新的梯度计算、重新训练、BN校准或正式再评价。
