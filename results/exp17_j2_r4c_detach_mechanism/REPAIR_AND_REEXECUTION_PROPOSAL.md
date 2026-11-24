# R4-C最小诊断修复与重新执行提案

`NOT_AUTHORIZATION / NO_NEW_EXECUTION`。原v1目录、授权、源码、12次CPU调用及1A+1N消费保持不变，pending不得补写。

1. 将三种计数分开：根Module hook应观察一次dict入口；单独包装诊断副本的forward方法观察dict/tensor两次方法分派；实际A 0–23与N 0–16层序继续严格检查。不得简单删掉遍历检查。
2. 在任何完成断言之前保存实际计数、输出身份、状态检查和资源快照；失败回执列出expected/observed。保持图等价性门与六次梯度顺序不变。
3. 对临时目录清理的相对路径，依据dir_fd解析真实目标，仅允许独立settings内的临时文件操作；不放宽原数据和证据目录保护。
4. 先做无Torch、无Autograd、无模型的分派纯桩测试：显式模拟Module调用触发hook，而直接self.forward不触发hook；再覆盖重复/错序/待提交计数、输出先封存和dir_fd边界。原桥与CPU Fixture逐字节/函数身份不变时复用本次12次验收，不自动追加CPU反传。
5. 新入口引用原authorization SHA、ledger SHA和HOLD回执，独立记录继承消费与新增上限。旧A图已丢失，新对照必须重新构建两张图，不能把它称为原预算无损恢复。

| 操作 | 原v1已消费 | 提议新增 | 研究累计 |
|---|---:|---:|---:|
| A完整前向 | 1 | 2 | 3 |
| N前缀 | 1 | 2 | 3 |
| 模型组件Autograd | 0 | 6 | 6 |
| CPU小张量Autograd | 12 | 0，复用相同桥证据 | 12 |
| Optimizer / VAL / TEST | 0 | 0 | 0 |

原先2A+2N总上限不足以完成重新对照；必须明确批准上述增量后才执行。新运行仍限10GiB allocated、20分钟，固定完整Checkpoint/Projection/args、同一position1和原Pair资格。若桥数学实现需要改变，则不能复用旧CPU验收，另行说明新的CPU预算。

修复仅针对独立诊断工具；不改生产BN、G1、Runtime、Checkpoint、科研参数或正式结果。本提案没有应用到已冻结的v1源码，也没有重新启动。
