# Exp17-J2 R2：既有 TRAIN 诊断记录的独立工程解释

状态：TRAINING_EXECUTION_COMPLETE_RESULT_VALIDITY_HOLD。本报告只读既有结果；没有新增模型读取、forward、backward、更新、VAL、TEST或probe。该解释属于Codex工程复核，不代签外部独立科研审计。

**本批次已确认的近因是J2对BN运行统计的依赖造成train/eval归一化失配。** 同一权重、同一32图TRAIN输入、相同eval head，仅将BN改为不回写的当前batch统计，检测及loss显著恢复。BN统计为何漂移、具体层/mean/variance贡献和是否由Pair引起仍未决。

## 对照结果

32图含62个GT。固定conf=.25、IoU=.50类别一致一对一Box匹配：

| 既有记录 | 检测 | Box TP | FP | FN | conf=.001 TP |
|---|---:|---:|---:|---:|---:|
| j1_native | 24 | 19 | 5 | 43 | 54 |
| j2_native | 0 | 0 | 0 | 62 | 0 |
| j2_batch_stats | 24 | 18 | 6 | 44 | 52 |
| j2_ema_saved | 0 | 0 | 0 | 62 | 0 |

原J2 conf=.25在NMS前候选已经为0；raw class logit max=-4.84460449、mean=-60.3032913，max sigmoid约0.007809265。conf=.001仍有30个FP/0TP，不能笼统写为所有阈值均无预测。batch-stat后logit max=2.47515488、mean=-16.6127472；低阈值下GT最佳几何IoU均值由0.0080465848升至0.70944662（J1 0.73914932），所以改善同时涉及几何预测，并非单纯越过conf阈值。

这证明该固定TRAIN批次上存在可被BN统计替换解除的功能性失效；本次未计算Mask TP/AP或正式VAL，不能称为实例分割科研性能已恢复，更不能据此选择模型或改冻结推理协议。

## Foreground、loss与原训练曲线

| 当前FP32诊断 | P3/P4/P5 foreground | box | seg | cls | DFL | YOLO sum |
|---|---|---:|---:|---:|---:|---:|
| j1_native | 420/150/16 | 1.66458 | 3.10823 | 2.07415 | 2.07472 | 285.49408 |
| j2_native | 305/207/63 | 2.81695 | 423.165 | 17.5858 | 117.32 | 17948.41 |
| j2_batch_stats | 413/157/15 | 1.79132 | 3.29009 | 2.24745 | 2.18268 | 304.36942 |

原running-stat路径foreground总数575、batch-stat为585、J1为586，均非0。当前批次不支持“没有正样本/全head死亡”解释；也不能从这次assignment重建历史foreground或梯度。

YOLO sum≈32×原生parts总和（语义分量sem_loss为0）。原J2 seg、DFL、cls高于batch-stat约128.62/53.751/7.8248倍。此处parts是同一raw output复用criterion所得，未新增forward，但head/BN为eval诊断口径，且使用FP32。

原始训练e5/e10/e15/e100的YOLO均值分别为452.9315/417.1070/398.5491/280.6316，而这些checkpoint当前固定批次eval-stat YOLO sum分别为5.06965e8/599966.5625/4131586.5/17948.4102。训练态loss下降与推理态失效可以共存；这些不同批次、不同mode/精度的数据不能当作配对历史分量曲线。

原正式joint total=YOLO+8×SupCon+4×Pair（2100条sanitized step的最大浮点残差3.96371e-05）；e100 joint mean303.38794含aux，不能与batch-stat诊断YOLO304.36942混为同一loss。正式原始日志未存box/seg/cls/DFL及foreground历史，本次不能补造。

## 已检查的其他机制边界

- J1 full→native、J2 full→native、J2 native→saved、J2 fused→backend：5类记录tensor逐位一致。对应本输入的export/序列化/backend单独失配假设被明显削弱。
- J2 saved→fused与J1 native→backend均有数值差异和部分allclose marker失败，已保留；两种conf下的检测数与TP/FP/FN均未改变。不能把融合的marker失败单独当作根因。
- 两张冻结原TRAIN图含2 GT：J1 conf=.25为3det/2TP，J2为0det/0TP；支持失效不限于32图增强packet，但不足以排除所有输入分布因素。
- 同一32图上e5/e10均9600det=32×300、全部FP；e15为76det/0TP；final EMA也为空。异常不是简单的“全程都无预测”；early head输出及criterion分量已极端异常。仅final做了BN统计干预，不能宣称所有早期异常均由同一机制完整解释。
- 当前干预同时替换所有BN层的mean/variance使用方式，保持affine参数/head eval/输入/持久state不变；它没有定位具体层，也不是BN校准、训练或新checkpoint。
- 先前CPU观察到部分BN variance远高于J1，与本批次干预形成相互支持的证据；差异本身仍不能归因Pair。AdamW的104→105单步吻合不能泛化到2100步历史。

## 预算、可复现性与状态

ledger链及每个case SHA均核验：18次=15次真实TRAIN forward+3次CPU stride构造forward，全部COMMIT、无未用额度；backward/update/VAL/TEST均为0。v1路径allowlist失败发生在创建authorization/ledger及forward前，旧失败记录保留；成功v2未重放已耗调用。

成功记录保留4条cuBLAS nondeterminism warning：runner使用deterministic warn_only且未配置CUBLAS_WORKSPACE_CONFIG。当前逐位对照是实际观测，不能外推为严格重复运行保证；不为此重跑或扩大18次预算。

正式2100步训练/100epochs执行完成的事实保持；80个既有VAL pass不会被本次TRAIN诊断替换。科学收益、原始全0 AP的可用解释范围与后续修复均继续HOLD。尚待解释的是漂移形成过程和因果来源，而不是继续将全0 AP直接当作Pair方法的有效负结果。
