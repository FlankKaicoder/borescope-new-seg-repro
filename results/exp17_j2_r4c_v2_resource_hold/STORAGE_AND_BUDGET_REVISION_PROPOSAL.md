# R4-C v3存储与预算修订提案

`NOT_AUTHORIZATION / NO_NEW_MODEL_EXECUTION`。保留v1/v2全部原始记录和pending，科学参数、Checkpoint、完整Projection、保存args及position1不变。

## 固定工程方案

Graph A的Autograd保存张量改为独立目录下的无损磁盘缓存；Graph B继续CPU卸载。每个保存项记录dtype、shape、实际CPU副本stride、内容SHA和字节数，解包严格验证并恢复原布局。CPU小张量存储往返测试不调用模型或Autograd；原桥函数和CPU导数Fixture保持相同并复用验收。

缓存写入只发生在新诊断目录，不能修改或复制原始数据文件、写回Checkpoint或复用旧缓存身份。写完fsync，读入完成后释放CPU缓冲；Linux文件页缓存按本轮文件范围释放，避免页缓存耗尽容器余量。不得压缩精度、重算前向、拆batch或改变算法。记录逐次pack/unpack字节、活跃缓存量、CPU/GPU峰值。

固定门限：GPU allocated≤10GiB，RSS≤20GiB，容器余量≥2GiB；新增活跃临时磁盘≤16GiB，磁盘余量≥2GiB；数值进程≤1200秒。当前磁盘可用约18.84GiB。单图A实测RSS约13.19GiB为容量参考，不能作为新实现峰值保证。存储完整性或容量检查失败即停止，不自动改门限或重试。

仍先完成两图前向及状态等价验收，再执行YOLO A/B、SupCon8 A/B、Pair4 A/B六次Autograd。不得改成先对A反传再比较B，也不引入新的数学图定义。

## 追加范围

| 操作 | v1/v2已消费 | 提议新增 | 累计上限 |
|---|---:|---:|---:|
| A全网attempt | 3，其中1次部分执行 | 2 | 5 |
| N前缀attempt | 2 | 2 | 4 |
| 模型组件Autograd | 0 | 6 | 6 |
| CPU小张量Autograd | 12 | 0，复用 | 12 |
| Optimizer / VAL / TEST | 0 | 0 | 0 |

需要明确批准：新增上述数值调用，以及原计划未允许的最多16GiB无损临时激活磁盘缓存。原预算的未使用N名额不用于掩盖新运行；新账本引用两个原授权、账本SHA及资源HOLD，分别报告继承与新增消费。

预备验收包含：普通/非连续/带stride张量逐字节往返，缓存SHA及dtype/shape错误拒绝，资源预留和重复/未完成调用拒绝，释放文件页缓存的路径边界。不得消耗新的CPU Autograd；如需改动桥数学实现，应另行说明CPU导数复验预算。

此处仅提出可审查修订，没有创建v3数值授权或启动新的模型调用。
