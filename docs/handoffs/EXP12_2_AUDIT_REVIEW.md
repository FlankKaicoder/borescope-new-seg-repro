# Exp12.2 formal SSL audit handoff

状态：`PASS_WAITING_EXP12_3_AUTHORIZATION`。

## Confirmed facts

- Exp12.1 fixed-budget TRAIN-only SimSiam 完成 100/100 epochs、2,000 optimizer steps；不早停、不选 best，固定使用 epoch 100。
- 100 个 epoch 全部指标 finite；minimum feature std 0.1047977105，minimum embedding variance 0.9999025553；collapse 连续计数始终为 0。
- encoder 在正式首步获得 present/finite/non-zero gradients 120/120，并改变 120/120 参数张量。
- 最终改变 120/120 参数张量与 1,365,471/1,365,472 参数元素；全部 finite；BN buffer 变化另行统计为 120/120。
- final checkpoint 与 backbone export 均完成 120/120 parameter hash round-trip。
- final checkpoint SHA256：`478a5cd14f850925c8c6ba1a78459d7d93c0076511234a20b4c58f4de0a88c7b`。
- backbone export SHA256：`3ae6909d1d377ed18c8d532b7fde50cb0d64c092f4137dc035aff5a2a94f580c`。
- TRAIN exact-set 668；VAL/TEST access 0/0；下游未运行。

## Interpretation boundary

普通 SimSiam 的工程与 backbone-update Gate 已通过，但表示质量和分割收益尚未评价。不能用 SSL loss、参数变化幅度或 smoke/formal checkpoint 直接宣称 mAP 提升，也不能修改 Exp11 最终结论。

## Next allowed action

等待用户单独确认 Exp12.3 Route A/B 公平下游。获批后必须固定相同 split、epoch、batch、optimizer、scheduler、augmentation、seed 和 evaluator；Route B 只注入 encoder layers 0–10，并在注入后和 Trainer 构造后两次验证，Neck/Segment Head 相对 Route A 初始化变化必须为 0。只使用冻结 VAL 比较，TEST 禁止参与。Multi-scale 方法仍不在当前范围内。
