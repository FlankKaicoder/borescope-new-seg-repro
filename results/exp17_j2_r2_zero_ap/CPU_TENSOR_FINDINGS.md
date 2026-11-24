# Exp17-J2 R2 CPU 离线权重、BN 与 AdamW 轨迹

状态：CPU_OFFLINE_ANALYSIS_COMPLETE_RESULT_VALIDITY_HOLD。

25份checkpoint均以weights_only=True、map_location=cpu读取：20份每5epoch periodic、B0 104/105双槽、B1 2100双槽及冻结J1 Last。未构造模型/优化器，未做forward/backward/VAL/TEST；原输入读前/读后SHA全部一致。

20份MASTER/EMA轨迹输出按backbone、neck、cv2/cv3/cv4各尺度、proto、dfl及projection分域；BN逐层与AdamW逐参数另列。

## 主要发现

- J1/J2 Last的模型配置、类别、565个state entry、optimizer参数顺序及最终AdamW超参数一致。294个有状态参数的step与checkpoint update一致；冻结DFL无AdamW state。
- 大数值主要出现在BN running_var缓冲，不能称为训练权重爆炸。独立parameter-only表含295个参数张量（2,868,805元素，含16个冻结DFL元素）。
- J2 final MASTER全BN running_var L2为237348.177，J1为31437.437，比例7.54986。两者均有限，全部90层的num_batches_tracked均为2101；所有periodic记录未见负/零running variance。
- J2/J1差异最大的12层列在bn_j2_j1_comparison.csv；前6层的20个periodic轨迹列在selected_bn_layer_trajectory.csv。

| BN running_var | J2/J1 L2比 | J2最大值 | J1最大值 |
|---|---:|---:|---:|
| model.16.m.0.cv1.bn.running_var | 1919.5 | 19130.6133 | 5.50149822 |
| model.13.m.0.cv1.bn.running_var | 413.281 | 23954.4043 | 42.9380569 |
| model.5.bn.running_var | 296.265 | 78868.1953 | 195.013855 |
| model.13.m.0.cv2.bn.running_var | 277.664 | 26202.3125 | 59.0823021 |
| model.2.m.0.cv2.bn.running_var | 268.369 | 826.771301 | 1.52330351 |
| model.4.m.0.cv2.bn.running_var | 209.851 | 5759.31641 | 12.2908287 |

这些是后续BN/激活调查的线索。J1本身也有较大head/proto variance；在没有TRAIN对照前向的情况下，不能从上述差异确认BN根因，更不能归因Pair。

## 单步更新与双槽

104→105按m差分近似重建的post-unscale/post-clip梯度L2为9.99999981951，FP32舍入L2保守估计约9.867e-06；v差分的g²总和为100.000114817，未见超出舍入估计的负g²。
实际Δθ L2=1.04550016469；使用保存的m/v和精确超参数，理想AdamW公式残差L2=4.97102e-06；独立FP32张量表达式复现残差L2=5.07088e-07，2823311/2868789元素逐位相同。近零值的最大ULP差可大（3462），绝对误差仍小，不能只按ULP判错。
这仅检查1个连续更新。moment差分不是历史原始preclip梯度，不能泛化为2100步全部梯度正确。

B1两个槽均为2100；state/ema/optimizer/scaler/scheduler/rng/audit/counters/sampler/training_modes全部相同，仅extra不同，支持AUX未改变训练tensor。没有2099→2100两端状态，不能推导最后一步梯度。

## 证据边界与文件口径

- domain_statistics.csv及state_deltas.csv包含所有浮点state（含BN缓冲）；parameter_domain_statistics.csv仅含参数。相邻parameter-only delta未单独保存，不能用两个L2总量相减补造。
- moment adaptive/decay项是代数proxy，不是新更新、实测next-step或因果对照。
- BN affine/buffer统计不提供激活分布、运行时epsilon、foreground分配、原生box/seg/cls/dfl历史分量或检测logit。
- finite、参数变化和单步公式吻合不能证明预测有效。保持TRAINING_EXECUTION_COMPLETE_RESULT_VALIDITY_HOLD；不据本分析改协议、超参数或checkpoint选择。
- 所有公开形状报告只使用相对标签；input_inventory.private.json单独保存绝对来源与SHA。
