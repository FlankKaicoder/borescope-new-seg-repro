# EXP18 V2 CPU 交付证据

执行入口为 `python -B -m unittest tools.exp18.tests.test_storage_v2 -v`。使用 Python 3.12 和 Torch 2.8.0+cpu 时，可将环境变量 `EXP18_V2_MEASUREMENT` 指向新的 JSON 路径，生成序列化字节和恢复轨迹回执。测试的私有产物落在本 worktree 的 `private_assets/exp18_storage_v2`。运行需要可用的 Torch CPU wheel 与 NumPy。

| 产物 | 用途 |
|---|---|
| [V2 测试日志](CPU_TEST_LOG.txt)、[轮换故障日志](CPU_ROTATION_FAULTS_LOG.txt)、[测量回执](cpu_measurement.json) | H200 合成连续和恢复对照，状态哈希、事件计数、临时峰值和实测文件大小 |
| [选定回归](CPU_REGRESSION_LOG.txt)、[守卫回归](CPU_GUARDS_LOG.txt)、[真实入口拒绝](V2_NATIVE_GATE_LOG.txt) | 旧数学与事务回归、CPU/原始资产/TEST 屏障、V2 配置拦截 |
| [容量回执](capacity.json) | 历史字节与合成测量、三臂估算及正式容量缺口 |
| [伙伴详情](coverage/partner_diversity_by_anchor.csv)、[按类汇总](coverage/partner_diversity_summary_by_class.csv)、[伙伴边](coverage/partner_edges.csv) | 521 个 K anchor、497 个 S anchor、6,265 条聚合错配边 |
| [联合覆盖](coverage/coverage_class_version_p3_cross.csv)、[账本校验](coverage/coverage_receipt.json)、[独立表对照](coverage/reference_reconciliation.json) | 4,200 个封存元数据记录、四种投影的 145 行、此前 52 个正 K 单元逐项对照 |
| [SHA 清单](sha_manifest.json) | 输入、输出与本次源文件 SHA |

按类统计的辅助曝光分母是该类 S 曝光。一个伙伴的 18 个 anchor 占 497 个 S anchor 的 3.62%，这组贡献 95 次曝光，占全部 22,437 次 S 曝光的 0.42%。18 个中有 3 个只出现一次，15 个重复出现仍只观察到一个伙伴。

| 类别 | K 唯一 A | S 唯一 A | S 曝光 | 单伙伴 A |
|---|---:|---:|---:|---:|
| Burn | 54 | 46 | 1,701 | 3 |
| Crack | 52 | 49 | 2,968 | 2 |
| Dent | 94 | 91 | 5,373 | 5 |
| Material missing | 80 | 77 | 4,747 | 3 |
| Tears | 40 | 39 | 2,160 | 2 |
| Tip curl | 0 | 0 | 0 | 0 |
| corrosion | 201 | 195 | 5,488 | 3 |

覆盖分析只读本地 CPU 归档和已接受的步索引。CSV 空值表示无分母或该维度不适用。真实 TRAIN 像素回放、真实模型、CUDA、真实训练、VAL 和 TEST 均为 0。CPU 合成的评价回调与容量测量不代表真实评价或 CUDA 恢复通过。
