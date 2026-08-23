# Exp12.1-S smoke review handoff

状态：`PASS_WAITING_FORMAL_SSL_AUTHORIZATION`。

## Confirmed facts

- 冻结 TRAIN exact-set 668；VAL/TEST 访问均为 0。
- 框架加载 encoder 时 `requires_grad=True` 为 0/120；显式解冻后为 120/120。
- optimizer identity：encoder 120/120 恰好出现一次；全模型 132 个参数对象均恰好一次；无 duplicate。
- 第一个真实 TRAIN batch：120/120 gradient present、finite 且 non-zero；step 后 120/120 参数变化，`changed_ratio=1.0`。
- 1 epoch：20 steps、640 processed samples；120/120 参数变化，1,365,467/1,365,472 参数元素变化；全部 finite。
- `feature_std=0.1047977105`、`embedding_variance=0.9999025553`，smoke 未出现 collapse 信号。
- checkpoint 和 backbone export 均与内存 encoder 完成 120/120 parameter hash round-trip。

## Interpretation boundary

Exp12.1-S 只证明普通 SimSiam 工程链路能够真实更新 backbone。它不证明表示更优、不评价 mAP，也不能替换 Exp11 Baseline 或修改最终 TEST 结论。smoke-only 权重禁止用于正式下游。

## Next allowed action

等待用户单独确认 Exp12.1 正式 TRAIN-only SSL。正式训练必须使用新 `formal_<run_id>` 目录，逐 epoch 记录 loss/feature mean/feature std/embedding variance，并在结束后执行内存、checkpoint reload、backbone export 三重参数审计。`changed_ratio == 0` 时仍必须标记 `INVALID_BY_BACKBONE_NO_UPDATE` 并停止；VAL/TEST、下游和 Multi-scale 方法仍禁止。
