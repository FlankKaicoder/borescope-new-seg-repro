# EXP18 完整 CPU 回放的独立元数据验收

`full_replay_audit.py` 是冻结 a10 执行快照以外的审查工具。它不修改执行源码，不读取图像或 `.pt`，不构造真实模型，不初始化 CUDA，不运行 VAL/TEST。

完整回放及 `tools.exp18.coverage_report` 均成功后，在原服务器同一固定 Python 环境中执行：

```bash
/root/autodl-tmp/borescope-new-seg-repro/.venv/bin/python -B full_replay_audit.py \
  --code-root /root/autodl-tmp/exp18_cpu/20261125T131451Z/code_a10 \
  --replay-root /root/autodl-tmp/exp18_cpu/20261125T131451Z/full_a10/replay \
  --coverage-root /path/to/new/coverage \
  --output /path/to/new/full-audit
```

`--output` 必须是新目录，且位于三个不可变输入目录之外。脚本不清理失败输出；失败生成 `failure.json`，通过生成 `summary.json` 与 4,200 行 `steps.csv`。

审查包括：

- 恰好 200 个连续 COMMIT、4,200 批、133,600 张图曝光、253,200 个原始实例曝光；各轮668来源不重不漏。
- COMMIT、压缩包、绑定、Protocol、输入参考、原生源码及依赖版本；读取后再次核对元数据/源码 SHA。
- 六张量 SHA 重新组成原始批 SHA；前2,100批重新比较冻结J1/R5身份及trace SHA；后2,100批明确历史不适用。
- 所有4,200批的增强目标逐行通过原始native label ordinal映射回冻结TRAIN类别、Polygon、来源组和JSON ordinal；池化及零格点行覆盖全部增强目标。
- 独立 0/1 方阵最大指派：对角丢弃代价1、合法非对角0、非法边K+1。使用固定SciPy求解器重新验证每批闭合最大人口，不调用执行代码的 `build_matching` 或 `legal_pair`。
- 记录的共同S、无固定点、唯一N、类别/修复族/来源组/面积/格点约束、精确Fraction权重边际、自己ROI的稀疏N索引及多Polygon共享一次N图像回放。
- 独立 Decimal H200/H100 日程与源SHA绑定的冻结数值helper逐点比较；每轮seed字节独立核对。这里只验证配置数学，不声称运行过真实优化器。
- 覆盖报告全部4,200批最大匹配重算、表格SHA、完全相同的COMMIT集合，以及完整guard receipt。
- K/S全局Polygon、来源组、来源图像的并集人数；不把每批unique数相加。逐类人口引用已验证SHA的覆盖表。

脚本主要使用标准库；冻结日程helper的warmup使用NumPy，独立最大人口证明使用固定SciPy。两者不导入Torch/Ultralytics，求解器二进制SHA写入报告。覆盖工具还应重算带种子次级代价的具体地图，二者分开验证。

限制：原始像素及完整增强trace不重新生成，SHA代表冻结CPU回放时采集的证据。Python guard不是操作系统层的系统调用证明。此脚本不重新运行83项回归测试，不替代1899输入/446源码的前后核对或科研独立审查，不授予GPU、λ或正式训练许可。

本脚本首次交付时仅完成静态检查，尚未在完整4,200批账本上运行；不得把脚本存在或静态通过写成实验验收通过。
