# EXP18 CPU验收完成｜GPU阶段待Gate

**2026-11-28，Asia/Shanghai。当前状态：`EXP18_CPU_READY_GPU_PENDING`。**

提交GPT独立审查的完整材料见[审查报告](EXP18_CPU_GPT_REVIEW_REPORT_20261128.md)，含各类覆盖分母、尺度/版本偏移及GPU入口与容量问题；本交接的原验收结果保持不变。

用户开启服务器后，本轮完成此前延期的全量CPU数据回放、独立元数据审查和E盘归档。真实模型构造/加载/前反向、CUDA初始化、GPU计算、VAL、TEST均为0。λ仍为`null`，200 Epoch正式训练未授权。按原授权在CPU READY停止，等待ChatGPT独立审查与下一阶段Gate。

## 身份和代码

- EXP18工作树：`E:/github_repositories/borescope-exp18`；分支`codex/exp18-cpu-engineering`。
- 初始HEAD：`3d58386d969a56279660d535bb63ce87e2255292`；运行源码提交`6e3955e1c2632e1d832631935454397fba3a82ff`，审查工具提交`820537c97bf8e625e7fc94d01c171bc2b72cb518`。
- 冻结执行快照：服务器`/root/autodl-tmp/exp18_cpu/20261125T131451Z/code_a10`。446文件前后SHA一致，历史R5工作树未改动。
- source.tar SHA：`db2a7f79ce9bd1f92ca98251cd427de4822d10d15e91306d5adb88a3f7708b58`；manifest SHA：`d545d9e59232aa60c5109fac7128ba43dfbb0207cd85f9614ec44e823d48408e`。
- Protocol SHA：`fab9e2289a96fab68e90c8560548de2f2849b9c3d9f6a844215d5a8e7b86e15b`；完整执行绑定SHA：`68aa155ea740173b58ea3a1e612f29c4828fb41f29d80b796697da5016b5c5ac`。

## CPU验收结果

| 项目 | 结果 |
|---|---|
| 原生TRAIN回放 | 200连续COMMIT、4200批、133600图次、253200原始实例次；唯一TRAIN图仍为668 |
| 历史对照 | 前2100步与J1/R5的原生数据、增强、实例及身份字段精确一致；后2100步历史不适用 |
| 配对审查 | 4200次带种子具体地图重算；另用独立0/1方阵验证4200次最大闭合人口；权重与N own-ROI索引通过 |
| P3与身份 | 真实增强后几何记录、Valid完整支持、80×80覆盖几何及全部增强目标原始ordinal/class/group绑定通过 |
| N图像回放 | 20470次CPU图像重放，1967次多Polygon图像复用；没有真实模型前向 |
| 空匹配 | 3901批S非空、299批S空；最长连续空段为step1–3。非空仅表示存在辅助监督，不等于实际更新或非零梯度 |
| 学习率与seed | 独立Decimal验证H200/H100共6300日程点、200个seed轨迹；属于配置数学，未执行真实优化器 |
| 已有CPU测试 | 83项PASS与固定TRAIN包1/7/11/21按原SHA复用；本轮未重跑；冻结源码和环境源码前后无漂移 |
| 资产保护 | 1899输入的大小、mtime和SHA前后不变；446源码、原生依赖源码和旧测试证据SHA通过 |
| 独立技术终验 | 源码静态审查、实际回执/4200索引和E盘包独立复核PASS；科研Gate仍待审查 |

## 实际辅助人口

这里的K/S唯一人数是完整200轮的并集；曝光为跨轮重复次数。每类来源组数不能相加代替全局独立组数。

| 类别 | K唯一Polygon | S唯一Polygon | K组 | S组 | K曝光 | S曝光 |
|---|---:|---:|---:|---:|---:|---:|
| Burn | 54 | 46 | 32 | 29 | 10613 | 1701 |
| Crack | 52 | 49 | 39 | 36 | 10268 | 2968 |
| Dent | 94 | 91 | 84 | 81 | 18439 | 5373 |
| Material missing | 80 | 77 | 77 | 74 | 15572 | 4747 |
| Tears | 40 | 39 | 38 | 37 | 7783 | 2160 |
| Tip curl | 0 | 0 | 0 | 0 | 0 | 0 |
| corrosion | 201 | 195 | 72 | 72 | 38192 | 5488 |

全局K：521 Polygon/327组/345来源；S：497 Polygon/316组/334来源。K曝光100867、S曝光22437，曝光保留率22.24%，唯一区域保留率95.39%。Tip curl仍参与全部七类主任务，没有直接A1配对监督。没有放宽caliper、版本族或来源组约束。

## 资料和复核入口

- [完整验收回执](results/exp18_cpu/full_replay_20261128/acceptance_receipt.json)、[结果入口](results/exp18_cpu/full_replay_20261128/README.md)、[交付记录](results/exp18_cpu/full_replay_20261128/delivery.json)。
- [4200步数据/匹配SHA索引](results/exp18_cpu/full_replay_20261128/full_audit/steps.csv)、[辅助活动表](results/exp18_cpu/full_replay_20261128/cohort_profile/activity.csv)、[分类覆盖表](results/exp18_cpu/full_replay_20261128/coverage/population_by_class.csv)。
- 本地总入口：`E:/autodlwenjian/exp18/cpu-20261128/README.md`。`core.tar.gz`为215898884字节，SHA `0e4bd57cb5778f09f4de5c723e504199e0531b65a739f3711223385056b33463`。
- 包含469成员/452 regular；全部成员字节与属性、完整gzip CRC/ISIZE、4个隔离样本恢复通过。IPC socket列明排除；未删除服务器文件。`files.csv.gz`可按原服务器路径查找。
- 已测a10源码、继承11文档和旧测试完整证据沿用`E:/autodlwenjian/exp18/cpu-20261125`，不覆盖旧包。
- 服务器完整账本在原根`full_a10/replay`；本轮控制、审查和资产复核在`resume_20261128_01`。完成账本供后续后端只读使用；不需要重复执行已完成的旧resume命令。

## 成本与后续边界

容器本轮20逻辑CPU配额/30GiB，CPU回放进程4750.91秒（79.18分钟），5秒采样的cgroup峰值约1.50GiB；具体地图重算35.96秒、独立审查55.58秒、末端检查1.99秒。CPU主线程/两个worker与历史规则保持一致。

下一步先审查实际覆盖和方法假设，再另行冻结GPU协议、操作预算与容差。真实YOLO初始化/前反向、AMP、connected BN真实统计梯度、clip、原生恢复/导出等价、TRAIN-only λ标定及smoke全部待开卡Gate。现有CPU数据回放不能代替这些验证。

正式训练仍需容量Gate：append-only实现每臂计划4401个完整恢复状态，三臂13203个；应以真实smoke测得的单状态字节数核算存储，不能把剩余约45GiB自动视为足够。GPU耗时/显存尚未实测。不运行VAL调λ或接触TEST，不自动追加seed，不自动启动三臂训练。
