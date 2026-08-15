# Exp11 Evaluation-Invalidating Gate

状态：`STOP_WAITING_USER_CHATGPT_REVIEW`。

Candidate Freeze 已在任何 TEST 尝试前完成并提交：`9991fcfcb9cf6c0ab8920ad7deadeed579ce5585`。冻结方法、seed、checkpoint 和 SHA 均未改变。

首次 Exp11 启动时，客户端 SSH orchestration 使用了不足的返回超时，约 5 秒后终止远端 evaluator。服务器随后确认：

- `exp11_final_test.py` 进程数为 0；
- `summary.json` 不存在；
- `overall_metrics.csv` 不存在；
- 没有正式 TEST metrics；
- 仅保留 Candidate Freeze 副本、command、environment 和 checkpoint SHA 四个前置文件；
- 没有发现 checkpoint mismatch、split mismatch 或 evaluator code bug；
- 无训练、threshold tuning、seed/checkpoint/method selection。

由于无法证明中断前 TEST dataloader 完全未被触达，`test_access_state` 保守记录为 `POSSIBLY_ACCESSED_BY_INTERRUPTED_ATTEMPT`。不得静默删除输出目录后重跑，也不得把重跑自动称为同一次访问。

继续所需的唯一人工决定：是否明确授权对**完全相同的冻结 seed44 Baseline checkpoint、相同 SHA、相同已提交 evaluator、相同固定参数**重试一次。若授权，必须先保留当前 partial 目录并使用新的 retry 输出目录；若不授权，则项目以 TEST metrics NOT_AVAILABLE 收口。

## 人工裁决

用户已于 `2026-08-15T19:35:06+08:00` 明确批准一次重试。授权范围固定为：相同 seed44 Baseline、相同 checkpoint SHA、相同 evaluator、相同参数；输出写入 `results/final_test/exp11_retry1`，原 `results/final_test/exp11` 保持不动；不允许第二次自动重试。
