# R4-E：Epoch5零AP是否具有同一BN Mean近端机制

`PROPOSAL_ONLY / NOT_EXECUTED / NO_MODEL_BUDGET_ISSUED`。

目标：将最早的正式零AP状态与终点的BN Mean功能失配连接起来。只选择Epoch5，不扫描其他epoch、BN子集或置信度。

## 固定资产与口径

- 原Epoch5完整105步B1 periodic Checkpoint，包含Projection；文件SHA继承R3输入表，运行前按原值复核，不替换成原生Last导出。
- 已保存args.yaml、类别/stride/runtime属性，以及position1的同一32张TRAIN／62增强后GT封存packet。原Protocol/Runtime/521 Manifest均不改。
- 独立模型副本，eval head；FP32/TF32关闭、固定RNG、冻结生产代码。权重与buffers从原文件只读恢复。
- 沿用R4-C已验证的无dummy-forward骨架恢复方式，构造阶段阻断Module前向；严格权重/七类/属性检查。只运行原生分割网络，Projection不调用。

## 唯一数值顺序与预算

| 操作 | 上限 |
|---|---:|
| A全网：原running统计 | 1 |
| A全网：所有BN Mean使用当前A批次值，Variance仍为原running值 | 1 |
| N前缀／Autograd／Optimizer Update | 0 |
| 构造器stride或synthetic forward | 0 |
| VAL／TEST／训练／校准 | 0 |

两次前向共64图像曝光，使用两份独立副本。Mean-only使用R3-B已验证的functional BN方法，前向内替换统计值，不写持久buffer、不保存校准模型。不得使用training-mode head作为对照。

只读保存输出、90BN输入/输出统计、原buffers、模型/模式/RNG身份，复用既定conf=.25和.001描述性Box匹配。NMS、IoU、类别映射不变；报告TP/FP及已有候选几何，不计算新的正式VAL/AP，不将结果混入80次正式pass。

## 验收与停止

输入、权重和原buffer身份一致；Mean-only与原running允许前向不同。只允许计划中的统计来源差异。每次先预留attempt、后提交结果；异常、额外前向、NMS超时、身份差异或资源超限即停止，不自动重试。

GPU allocated≤10GiB、RSS≤20GiB、≤20分钟；新增派生输出≤256MiB、磁盘余量≥2GiB。独立目录和账本，不使用磁盘激活溢出、不清理R4-C v3缓存。CPU只复用已有保护和统计替换验收；若修改必要恢复适配，补充不消耗模型前向的结构/身份测试。

原running若已能产生正确检测，应先解释TRAIN与正式VAL差异，不能预设需要“救活”。原running为0而Mean-only正TP，支持早期与终点共有Mean依赖；Mean-only不恢复也不能排除variance/联合/局部机制，但不追加扫描。两次完成即停止。

后续Epoch5的Detached/Connected梯度对照是另一份2A＋2N＋6 Autograd范围，需另行确定存储与预算，本提案不包含它。任何一种结果都不能还原Epoch1–4起点或证明Pair是唯一原因。
