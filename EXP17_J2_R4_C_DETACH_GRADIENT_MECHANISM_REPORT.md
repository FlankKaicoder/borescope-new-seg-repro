# Exp17-J2-R4-C：Detach梯度诊断执行HOLD交付

日期：2026-11-24（Asia/Shanghai）。依据R4-B提交 `7244bd361e2522649a6ec2a0cfa6fc94da063b0d` 和用户对完整R4-C计划的明确实施指令。

状态：`EXP17_J2_R4_C_HOLD_DIAGNOSTIC_COUNTER_MISMATCH`。
科研状态仍为 `HOLD_TRAINING_ORIGIN_UNRESOLVED`。

**本轮没有得到真实J2的Detached/Connected梯度比较。** 六组CPU导数验收通过；Graph A的一次完整A和一次N前缀返回后，新增诊断工具的调用计数断言失败。Graph B及六次模型组件Autograd均未执行，遵守失败停止规则，没有自动修复后重试。

## 1. 已完成的实现与准备

独立代码位于 `diagnostics/exp17_j2_r4c/`，没有加入生产 `tools/` 源码清单。完整2100步Checkpoint、Projection、保存args、Criterion、可训练参数名单、类别和模式通过恢复检查。恢复工厂绕开标准构造器的stride dummy forward，没有使用缺少Projection的原生导出权重。

完整Checkpoint文件SHA保持 `79b237241517e9112687f9ecd155d21c096ce54243e9a8267b21a963d58c675e`；保存args SHA保持 `d1dfef12c7d5808e8530443f84577eee40d09f42d01cffc47214efa2a42c495a`。正式Protocol、Runtime、Manifest、COCO、授权、Claim、Journal及80次VAL原记录均未改写。

固定position1仍为32张TRAIN、62个增强后GT。原资格和Replay实现准备出16个有效A/N Polygon、14个独立A来源组和14张N；B来自该原生batch的真实区域，没有修改采样或扩展N池。按原SHA解码14张N及对应Valid，未读取VAL/TEST图像。

诊断授权仅工程记录用户的“Please implement the plan”，没有伪称新的外部签名；原正式训练及VAL授权未重新签发。

## 2. CPU数学验收

六组Fixture覆盖一般单层、不同A/N数量、零梯度和零方差边界、多层共享无bias映射、大均值偏移、inplace SiLU和保存张量卸载。每组两次Autograd，合计12次，预算已消费。

梯度桥forward保留base输出，backward只追加A Mean/Variance余切。与显式connected参考的最大绝对梯度差：float64约 `8.88e-16`，float32约 `1.19e-7`。六组独立float64中心差分检查通过。FP32案例同时使用独立NumPy闭式参考验收；原CPU记录的 `finite_difference_max_abs` 对案例5/6是FP32梯度与float64差分的描述性差值，不是该闭式参考验收的误差字段，不能据此误读阈值。

这些证据只验证小张量梯度桥。真实网络中多层统计传播、Pair梯度大小及共享参数变化仍未验证。预定的根入口分派纯桩测试没有在GPU前充分落实，导致计数缺陷漏检，这是本轮诊断实现的不足。

## 3. 实际调用与停止位置

| 项目 | 实际 | 状态 |
|---|---:|---|
| CPU Autograd | 12 | 六组PASS，全部提交 |
| Graph A完整A遍历 | 1 | 提交 |
| Graph A的N 0–16遍历 | 1 | 已返回，但计数门失败，保留pending |
| Graph B A/N | 0 / 0 | 未开始 |
| 模型组件Autograd | 0 / 6 | 未开始 |
| Optimizer / VAL / TEST | 0 / 0 / 0 | 禁止项未执行 |
| 自动重试 | 0 | 没有换目录重启 |

29行哈希链包含14个attempt、13个commit及HOLD。最后一个pending仍为 `A_N_prefix`，没有补写成功回执。原traceback到达 `forward_joint` 返回之后的 `finish_forward`，且此前N层序断言已通过，因此可以关联其已完成一次N前缀；这不等于该attempt已通过完整验收。

峰值allocated为1,278,415,360 bytes（约1.19GiB），reserved约1.56GiB。数值执行区间12.69秒，进程21.75秒。RSS峰值未封存，不能填入估计值。这些数字只覆盖失败前的单图路径，不能推断完整两图及反传资源需求。

## 4. 诊断自身的故障定位

冻结框架的 `BaseModel.loss` 使用 `self.forward(batch["img"])` 直接递归调用方法。该调用绕开 `Module.__call__`，不会再次触发根模块forward pre-hook。

本轮runner却要求根模块hook记录等于 `["dict", "tensor"]`，把方法分派与模块hook事件混为一谈。这个错误属于新增诊断工具，不是生产J2的导出、BN或梯度错误。

必须区分证据等级：原始记录保存了失败位置，但没有保存断言两侧的完整计数值。方法调用机制来自绑定框架源码与冻结runner的事后静态关联，标记为 `POST_HOC_STATIC_CALL_CHAIN`；没有倒填现场计数。

另有独立的进程退出清理问题：文件审计器没有按 `dir_fd` 解析相对unlink路径，误拒绝了独立settings内的临时文件清理。该问题发生在HOLD之后，未触及历史文件，也不是本次前向门失败的原因。

## 5. 封存、限制和下一步

634个受保护原文件前后SHA相同，生产Source Catalog仍为 `3ccdfe26b414c9c5c6c45a1be93a9fa38210cae8c36bc7e1a8a8c227a7b30f42`。原Last MASTER Mask AP=0、原80次正式VAL以及R3/R4-B结论保持不变。GPU进程已退出，设备占用回到约1MiB。

私有封存包共16个文件、55,629 bytes，SHA `6554fbfb19eff7d26d5b591f03e74342020b6819383b21f341fc95072a0cd982`。本地下载后逐文件SHA及29行账本哈希链复核通过；代码与执行版本逐字节一致。原始授权、路径清单、准备数据及traceback留在私有包。

公开交付：[执行摘要](results/exp17_j2_r4c_detach_mechanism/execution_summary.json)、[CPU验收](results/exp17_j2_r4c_detach_mechanism/cpu_tests.json)、[离线审计](results/exp17_j2_r4c_detach_mechanism/offline_audit.json)、[修复与预算提案](results/exp17_j2_r4c_detach_mechanism/REPAIR_AND_REEXECUTION_PROPOSAL.md)。

Graph A的计算图已随进程退出释放，不能只执行剩余Graph B来完成比较，也不能把本次A重跑当作未消费预算。后续需要独立审查最小诊断修复及新增2A+2N、6次组件Autograd范围；累计将达到3A+3N。原12次CPU导数证据可以在桥实现及Fixture身份不变的前提下复用。当前没有签发或执行这个追加范围。

本轮尚不能回答detach对真实J2 Pair梯度影响有多大，更没有新增证据证明它造成100 Epoch的Mean漂移。此处是诊断工程HOLD，不能当作新的科研负结果。
