# Exp17-J2-R4-C v2：计数修复通过与主机内存HOLD

日期：2026-11-24（Asia/Shanghai）。依据提交 `5b858ea2d0f6aa178dfa40b025782865fac948ac`，用户在明确追加2A+2N及6次组件Autograd范围后回复“好的，继续进行”。

状态：`EXP17_J2_R4_C_V2_HOLD_HOST_MEMORY_LIMIT`。科研状态仍为 `HOLD_TRAINING_ORIGIN_UNRESOLVED`。

**上次的计数问题已修复并通过真实Graph A验收；本次在构建Graph B时触及主机内存上限。尚未执行任何模型组件Autograd，因此还没有真实Detached/Connected梯度对照。**

## 1. 修复及CPU证据继承

独立v2将根模块hook、forward方法递归和实际特征遍历分别计数；输出观察先落盘再验收。临时清理按dir_fd解析真实路径，仅允许独立settings内操作。19项纯Python测试在本地及绑定服务器通过，没有导入Torch、构建模型或调用Autograd。

原梯度桥函数与六组Fixture身份保持相同，复用v1已通过的12次CPU Autograd，新增CPU Autograd为0。桥函数SHA `06507576c8b0170d095296a036794e820bd257aa155607012be3d5384a12db63`。旧v1目录、源码、授权、29行账本和未提交N attempt均未改写。

一次部署前Windows/POSIX路径转换错误在创建v2目录之前被拒绝并修正，未发生数值调用；这不属于数值重试。随后唯一一次数值进程没有自动重启。

## 2. Graph A已经确认的结果

完整2100步MASTER、Projection和保存args恢复后，固定32张TRAIN/62个增强后GT对应16个合格Pair及14张N。图像和Pair资格没有改变。

| 观察 | Graph A |
|---|---|
| 根Module hook | 1次dict |
| forward方法分派 | dict、tensor各1次 |
| A特征遍历 | 0–23恰好一次 |
| N前缀遍历 | 0–16恰好一次 |
| Projection | A/N各一次 |
| N前后BN buffers | 相等 |
| 参数、模式、RNG | 前向验收通过 |

FP32诊断损失：YOLO `304.3694152832031`，SupCon原值 `3.067108631134033`，Pair原值 `0.052910733968019485`，加权总损失 `329.117919921875`。SupCon8与Pair4的定义未改变。

这些数值说明该固定批次的正式原路径可计算有限且非零的Pair Loss。它们不是梯度证据，也不是新的VAL结果；与历史AMP probe不要求数值逐位相等。

## 3. 新的资源阻断

为了在任何模型反传前完成两图前向等价验收，执行器同时保留Graph A并构建Graph B。保存张量采用逐次CPU卸载。

| 资源观察 | 实测 |
|---|---:|
| Graph A完成后的RSS | 14,161,891,328 bytes，约13.19GiB |
| 停止时RSS峰值 | 21,475,991,552 bytes，约20.001GiB |
| 固定RSS门限 | 20GiB |
| 停止时容器使用 | 28,919,959,552 bytes，约26.93GiB |
| 容器限额 | 30GiB |
| GPU allocated峰值 | 1,278,415,360 bytes，约1.19GiB |
| 数值区间／进程耗时 | 20.56秒／30.72秒 |

直接触发项是进程RSS上限，超过门限1,155,072 bytes；当时容器仍有约3.07GiB余量。没有CUDA OOM、操作系统OOM或数值异常报告。不能因为超过幅度小就提高门限：Graph B的A路径尚未完成，N路径还没有开始。

静态检查确认每次保存张量都调用 `tensor.detach().cpu()`，没有共享存储复用。同一CUDA张量多次被保存时可能产生重复CPU副本；本次未保存逐次pack存储身份，因此无法量化重复副本贡献，也不能宣称简单去重必然足够。

## 4. 消费、封存和停止边界

v2账本共8行：A全网和N前缀均完成提交；Graph B的A遍历在进入module19后的卷积保存张量处停止，`B_full`保留pending。Graph B N及六次模型组件Autograd均为0。原12次CPU调用只读复用。

合并v1/v2，实际消费3个A attempt（2个完整返回、1个部分执行）与2个N attempt（均返回，v1仍未提交），模型Autograd为0。部分执行也占用预算，不清零、不补写成功。

650个受保护原文件前后SHA相同，生产Source Catalog仍为 `3ccdfe26b414c9c5c6c45a1be93a9fa38210cae8c36bc7e1a8a8c227a7b30f42`。原Protocol、Runtime、COCO、Manifest、Checkpoint、授权和正式80次VAL不变；原Last Mask AP仍为0。设备已释放，退出清理stderr为空。

私有证据包22个文件、122,271 bytes，SHA `e8b57a64d70382618da922aec41c8bc5b222f5fb8d4c4995ba3a85f0bb8e573e`；下载后21项文件SHA及8行账本链均复核通过。诊断脚本与执行源码一致，旧G4未提交改动保留。

## 5. 下一步与科研限制

推荐单独修订存储方案：Graph A保存张量无损落入本轮独立临时磁盘缓存，Graph B仍使用CPU卸载；不改变前向、梯度桥、Batch、精度或先验等价门。保持GPU10GiB、RSS20GiB及20分钟边界，新增最多16GiB临时磁盘及至少2GiB剩余空间保护。需要先验证序列化往返、stride/dtype身份、损坏拒绝及资源边界，再签发新的数值范围。

这修改了原计划“不使用磁盘激活溢出”的工程约束，且两图已退出，不能自动重试。具体见[存储与追加预算提案](results/exp17_j2_r4c_v2_resource_hold/STORAGE_AND_BUDGET_REVISION_PROPOSAL.md)。截至本交付，该提案未执行。

当前可以确认计数修复有效、原Graph A有活跃Pair Loss；仍不能量化detach对实际参数梯度的影响，更不能将本次资源失败当作Pair导致BN Mean漂移的证据。
