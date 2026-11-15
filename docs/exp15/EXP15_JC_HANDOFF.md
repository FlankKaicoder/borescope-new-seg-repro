# Exp15 J-C seed42 正式配对训练交接

**`EXP15_J0_J1_FORMAL_TRAIN_COMPLETE_PENDING_VAL_GATE`。** Seed42 的 J0 和 J1 正式训练均通过独立配对审计。

## 执行身份与范围

从已审计提交 `2aba71a929d2cb593d0e32377ee24811e0e4dc8d` 提取训练代码。正式协议 SHA256 为 `b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca`。私有授权只覆盖 `EXP15_J0_J1_FORMAL_AUTHORIZATION` 的 J0 和 J1。两个实验臂独立从同一冻结 COCO 权重初始化，先验收 J0 才启动 J1。J1 使用固定 `λs=8` 和 `temperature=0.10`。正式执行使用 AutoDL 单张 RTX 3080 Ti 12 GiB，Python 3.12.3、PyTorch 2.8.0+cu128、Torchvision 0.23.0+cu128、CUDA 12.8、Ultralytics 8.4.117。

初次零更新预检发现 Windows Git 归档将协议 JSON 传输为 CRLF，导致字节 SHA256 与已审计值不同。该次预检日志和原传输副本均保留。随后使用 J-B 已留存且字节 SHA256 完全匹配的协议，重新验证源文件、校准和 Smoke 回执、授权、运行库、冻结 TRAIN 的 1,336 个原始图像和标签文件。训练代码及超参数未调整。所有训练路径均使用 TRAIN，节点和数据仅在实验独立目录产生新文件。

## 完整性与公平性

| 独立验收 | J0 | J1 |
| --- | ---: | ---: |
| 完整 epoch／有效更新 | 100／2,100 | 100／2,100 |
| 持久尝试／原生 AdamW 调用／AMP 跳步 | 2,100／2,100／0 | 2,100／2,100／0 |
| 原始 TRAIN 图像曝光 | 66,800 | 66,800 |
| 增强前／后实例曝光 | 126,600／123,019 | 126,600／123,019 |
| 原生优化器参数张量／辅助投影张量 | 291／0 | 291／4 |
| 峰值 CUDA allocated／reserved，GiB | 4.684／5.025 | 4.714／4.988 |
| 训练与逐步数值审计耗时，秒 | 908.94 | 965.19 |

逐轮核对了 668 个 TRAIN 成员、21 个批次和 28 张的尾批。2,100 批原图顺序、每图实例计数、所有增强输入张量 SHA256、原生优化器分组和逐步学习率完全配对。独立复算学习率的最大绝对差为 1.39e−17。两臂初始原生模型状态 SHA256 均为 `8e0479ecdae6dc60558ed44d5d62c01f8b61075514bc34c109295950603fc886`。另核对 J0 与历史 Exp14 E0 的 2,100 个批次账本及原生总 Loss，逐步一致。训练前后原始 TRAIN 字节及标注哈希均通过复验。J1 的完整运行耗时比 J0 多约 6.2%。

## 数值、梯度与 Checkpoint

两臂共 4,200 次逐步数值审计均确认 Loss、梯度、FP32 主参数和 AdamW 状态有限，每次有实际参数更新。两臂原生裁剪阈值为 10，2,100 步均触发裁剪。每臂第 2,000 次有效更新后 GradScaler 从 1 升至 2，终点 growth tracker 为 100，溢出和跳步均为零。J1 的四个投影张量在每步均有非零梯度，投影头和 Backbone 参数哈希均发生变化。共 101 个监测步确认 BN 按各臂原生前向更新，J1 区域辅助分支没有额外改写 BN。

J1 的 SupCon Loss 范围为 1.645–4.503。全部 2,100 批均有有效 anchor，每批至少 33 个，中位 56 个。七类在整个运行中均有覆盖。原生 overlap raster 对 10 个 Burn 实例曝光没有留下有效像素，按 J-A 已冻结规则仅从辅助池排除。两臂原生输入与实例曝光仍完全配对。

每臂保留第 5 至 100 轮每 5 轮的 checkpoint，以及首步工程验证快照。独立复核全部 42 个文件的 SHA256、FP32 模型和 EMA 哈希、AdamW slot 和 GradScaler 状态。末轮额外通过原生七类架构的严格 CPU 装载，J1 移除投影头后用合成输入验证原生前向。两个末轮文件的私有本地副本均与服务器 SHA256 一致。

| epoch100 对象 | J0 SHA256 | J1 SHA256 |
| --- | --- | --- |
| `last.pt` 文件 | `711f42a31cd97dca21d0f29bddfcf8c3a2963fba2e3c5a705cd2785701569285` | `902fbc1efa4693e2601c162c76fb3736b38036d7eb0473438fe84317bb7895f5` |
| 原生 FP32 主模型状态 | `15d3a329892aaa32703d2eabd364baba74376059cdc965653bef965ac15a72f4` | `1ffe60884850dd454bb7c5ffdddcd9e5c5d9163f4db5d3b5b40026d10a24c107` |
| EMA 状态 | `456f92bc1672ee2cae7d8f93b74c13111cccfa62d3a15968f6dd11c10adce240` | `182c503e6c86f2eb2ca1bdaa280fd8cbabf23baec3cb2da4473e9670c0db669f` |

## 独立诊断与异常

J1 启动前登记了第 25、50、75、100 轮的两个固定 TRAIN 批次。训练后在独立进程读入只读 checkpoint，逐探针恢复模型和 Buffer，核验输入字节并分别反传原生与 SupCon。独立进程执行零次优化器更新，checkpoint 前后 SHA256 不变。

| J1 checkpoint | 固定 TRAIN 批 1 的加权辅助／原生 Backbone 梯度 | 固定 TRAIN 批 21 的比例 |
| --- | ---: | ---: |
| 第 25 轮 | 22.1% | 12.1% |
| 第 50 轮 | 25.6% | 19.1% |
| 第 75 轮 | 110.0% | 27.3% |
| 第 100 轮 | 189.6% | 41.7% |

两个固定批次的中后期结果表明辅助强度随训练状态和批次明显变化。比例描述裁剪前单批梯度，正式权重始终固定为 8。训练进程和验收均通过。工程记录另保留首次协议换行预检失败以及首次独立审计预验证误用系统解释器的日志，均在相应环节纠正。两个训练进程均首次运行完成、退出码 0。私有审计归档 SHA256 为 `208997e48fa9603df092541ab62e20a8f353a00b8d59fca5f5ff2602f93404fe`。

等待独立 VAL Gate。两个唯一的后续评价输入为上表文件各自的 epoch100 FP32 master。VAL／TEST 模型推理均为零，J2 和额外种子更新均为零。

[配对审计](../../results/exp15_jc_formal_seed42/pair_summary.json) · [J0 摘要](../../results/exp15_jc_formal_seed42/j0_summary.json) · [J1 摘要](../../results/exp15_jc_formal_seed42/j1_summary.json) · [逐批脱敏账本](../../results/exp15_jc_formal_seed42/matched_batch_ledger.csv) · [独立 TRAIN 梯度探针](../../results/exp15_jc_formal_seed42/isolated_train_gradient_replay.json) · [审计实现](../../tools/exp15/audit_jc.py)

[J-B 正式协议](../../results/exp15_jb/formal_protocol.json) · [J-B 校准交接](EXP15_JB_HANDOFF.md)
