# EXP18 无卡工程交付与后续验收

> 本文保留2026-11-25工程快照。2026-11-28完整CPU验收已完成，当前请读[CPU READY交接](EXP18_CPU_READY_HANDOFF_20261128.md)。

**2026-11-25｜`EXP18_CPU_ENGINEERING_PASS_FULL_REPLAY_PENDING`。尚未达到 `EXP18_CPU_READY_GPU_PENDING`。**

用户确认CPU资源固定，并允许等服务器有算力后完成困难测试。本轮停止全量回放及其worker，交付代码、现有验收、归档和Git；不自动进入GPU或训练阶段。

## 1. 身份与继承

- 独立worktree：`E:/github_repositories/borescope-exp18`；分支`codex/exp18-cpu-engineering`。
- 初始HEAD：`3d58386d969a56279660d535bb63ce87e2255292`。旧R5工作树和脏文档未被暂存、提交或修改。
- 继承文档11份，以原字节ZIP和实际SHA封存；[继承清单](results/exp18_cpu/provenance/inheritance.json)区分dirty/untracked与Git基点。
- 权威执行快照为服务器`code_a10`。446个文件全部匹配，83项测试后没有源码漂移。
- source manifest SHA：`d545d9e59232aa60c5109fac7128ba43dfbb0207cd85f9614ec44e823d48408e`。
- source.tar SHA：`db2a7f79ce9bd1f92ca98251cd427de4822d10d15e91306d5adb88a3f7708b58`。
- 源码提交和包级SHA见[交付记录](results/exp18_cpu/delivery.json)。

## 2. 已完成

三臂J1/A1-True/A1-Shuffled工程、Protocol、H200与独立H100配置、TRAIN-only加载与增强回放、P3几何、闭合最大指派、connected BN、空head冻结、双时钟恢复、不可变checkpoint/journal、跨进程lease、原生后端延迟工厂、逐图预测存储和原生导出代码均已构造。

| 验证 | 证据 |
|---|---|
| CPU回归 | **83项全部PASS**：数学、匹配、几何、BN、AdamW/EMA、恢复、日程、并发、路径/CUDA保护、预测存储 |
| 合成H200事件机 | 微型CPU图在第7步停止后恢复至4200步；不是真实训练 |
| 固定TRAIN包 | **1/7/11/21四包PASS**；文件SHA、真实数据加载和N回放 |
| 原生预演 | **21批/668图/1266原始实例**，六张量及trace精确匹配历史J1/R5 |
| 首轮辅助 | K502、S92、85次N图像回放；两worker均正常退出 |
| 末端只读校验 | **1899个输入、446个源码文件PASS** |
| 本地归档 | **4658个regular成员SHA通过**，4个成员样本恢复通过 |

真实YOLO构造/前反向、真实checkpoint张量加载、CUDA初始化、GPU计算、新VAL评价、TEST文件访问均为**0**。四个历史`.pt`是SHA白名单的TRAIN数据包，恢复测试只加载本轮合成图状态。

## 3. 实际人口——仅保存的首轮21批

| 类别 | K候选Polygon | S匹配Polygon | K组 | S组 |
|---|---:|---:|---:|---:|
| Burn | 54 | 13 | 32 | 10 |
| Crack | 51 | 13 | 39 | 13 |
| Dent | 92 | 23 | 83 | 23 |
| Material missing | 75 | 16 | 73 | 16 |
| Tears | 39 | 6 | 37 | 6 |
| corrosion | 191 | 21 | 72 | 18 |
| Tip curl | 0 | 0 | 0 | 0 |

全局K为502个Polygon/322组/338来源，S为92个Polygon/85组/85来源，S/K=18.33%。类别组数不能相加当全局独立组数。K有5次no-B，S为0；删除旧B代码依赖不代表曝光增加，七类主任务保留Tip curl。

此表只描述首轮，不能外推4200批。本地汇总核对闭合、约束及权重，未重算最大指派（`maximum_recomputed=false`）。[详细入口及范围](results/exp18_cpu/preview21_coverage/README.md)。没有为提升覆盖率放宽规则。

## 4. 资源、失败与延期

容器实测**0.5核CPU、2GiB RAM**，底层Xeon Gold6248；主机free不代表容器可用配额。多次进程收到外部SIGKILL(-9)，没有Python异常或内核OOM计数增长，不能断言已识别具体终止机制。

全部失败/partial保留。资源适配保持batch32、两worker、seed、增强、身份和匹配不变：每轮先真实读取21批并记录六SHA/ROI/trace/shape，退出worker后才N回放和配对；释放原图不以零图替代。原图SHA来自释放前真实数据，辅助后只复核保留的ROI/trace。

a09单轮预演退出0，约127秒。a10全量启动再次被终止，用户随后允许等待资源。**全量CPU账本0/4200批封存；21批预演不冒充全量进度。** GPU可以用于后续真实网络测试；原生数据加载、OpenCV增强、SHA回放仍依赖CPU，不能直接替换GPU实现后宣称字节一致。

## 5. 授权与待验

λ仍为`null`，200 Epoch正式训练未授权；H100是独立备用，不能运行中切换。R5科研Gate仍待独立审查。本轮未执行真实YOLO、AMP、显存/真实梯度、原生VAL或模型导出加载；当前CPU协议提前拒绝这些入口。

正式训练append-only容量Gate仍PENDING；所有epoch包和逐步恢复永久留存不能直接承诺45GiB足够，不能暗含删除/覆盖授权。[完整剩余与GPU清单](docs/exp18/GPU_PENDING.md)。

## 6. 包与恢复入口

本地：`E:/autodlwenjian/exp18/cpu-20261125/README.md`。`source.tar`/manifest保存已测试a10源码；`audit.tar.gz`保存服务器本轮regular产物、失败和合成恢复证据；`files.csv.gz`记录4658成员的原路径/包/大小/SHA。`context.zip`保存11份继承文档，`r5-history.jsonl.gz`保存2100步/49762ABN的身份投影。`archive_verification.json`记录包/成员/样本恢复校验。

服务器根：`/root/autodl-tmp/exp18_cpu/20261125T131451Z`。只使用`code_a10`，早期代码是历史工程证据。全部旧实验和原环境保留原处。

资源恢复并核对输入/源码/Protocol后，从冻结快照继续：

```bash
cd /root/autodl-tmp/exp18_cpu/20261125T131451Z/code_a10
python -B -m tools.exp18 replay --config configs/exp18/runtime_h200.json --output /root/autodl-tmp/exp18_cpu/20261125T131451Z/full_a10 --resume
```

恢复只跳过已提交且校验的epoch；本次会从epoch0新attempt开始，保留旧partial。完成200轮及全轨迹匹配重算、输入源码复核、独立CPU审计后，才能标记最终CPU_READY。

[实验设计与Gate顺序](docs/exp18/EXPERIMENT_DESIGN.md)、[代码与合同说明](docs/exp18/README.md)。没有自动GPU启动或定时任务；等待资源及下一阶段明确授权。
