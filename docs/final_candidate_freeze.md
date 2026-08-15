# Final Candidate Freeze（TEST 前冻结）

状态：**PASS / CANDIDATE_FROZEN_READY_FOR_EXP11**  
冻结时间：`2026-08-15T17:40:54+08:00`  
冻结时：`test_accessed=false`。

## Final method

最终方法为 **YOLO11n-seg Baseline**：official COCO `yolo11n-seg.pt` → supervised segmentation，`imgsz=640`、`batch=32`、`epochs=100`、AdamW、AMP=True、deterministic=True。

最终 checkpoint：

- seed：44
- VAL Mask mAP50-95：`0.3251567516159603`
- absolute path：`/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt`
- SHA256：`2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`
- size：6,007,012 bytes
- YOLO load：PASS
- state tensors：561/561 finite

选择规则是在任何 TEST 模型访问前，对三个独立 Baseline seed 的冻结 VAL Mask mAP50-95 做唯一一次选择：seed42 `0.29898123412927496`、seed43 `0.29950558246122316`、seed44 `0.3251567516159603`；因此冻结 seed44。TEST 后无论结果如何均不得换 seed/checkpoint。

## 为什么选择 Baseline

Baseline 是唯一完成三随机种子稳定性描述、训练协议清晰且不依赖未确认增益的方法。三种子 Mask mAP50-95 为 `0.307881 ± 0.014963`，并保留 Exp02 early validation AMP 的明确 numerical waiver。

Hard Mining 被拒绝：Treatment−Control delta 为 seed42 `+0.030354`、seed43 `-0.006673`、seed44 `-0.040311`，仅 1/3 为正，paired mean `-0.005543 ± 0.035346`。seed42 只能称 preliminary positive，不是稳定增益。

SupCon 只在 ROI classification representation 上 positive（macro F1 +0.015894），没有 segmentation improvement 证据。Stage2 为 negative。KD 被 engineering Gate 阻止且 NOT_EVALUATED；SimSiam reconstruction 未更新 trainable backbone parameters（0/120），downstream NOT_EVALUATED。

Uniform Control 的预定义角色是 training-budget control，不是 final candidate。其 mean 虽高于 Baseline，但 std 更大，禁止 post-hoc 升级；只作为 continued-training/seed-sensitivity signal。

960 resolution 不是欠跑实验。冻结 size audit 不支持简单的“目标越小 Recall 单调越差”，在有限时间下不足以推迟 Option A 项目收尾；状态保持 `NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE / FUTURE_WORK_ONLY`。

## Exp11 边界

Candidate Freeze 提交后，只允许对上述唯一 checkpoint 执行一次 frozen TEST evaluation：147 images / 285 instances。禁止 threshold/NMS sweep、checkpoint/seed/method comparison、训练或任何 TEST-driven tuning。第一份正式 TEST metrics 生成后，`MODEL_SELECTION_CLOSED=true`。

机器可读冻结证据：`results/final_test/candidate_freeze.json`。


## Post-TEST preservation record

Candidate Freeze remained unchanged. The selected seed44 checkpoint scored frozen VAL Mask mAP50-95 `0.3251567516` and final TEST Mask mAP50-95 `0.2716207089`. Checkpoint: `/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt`; SHA256 `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`. The `.pt` is excluded from Git; download it before releasing AutoDL.
