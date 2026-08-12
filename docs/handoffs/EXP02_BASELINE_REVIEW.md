# Exp02 baseline review handoff

## Frozen evidence

- Dataset v1 split hash: `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`;
- model: official `yolo11n-seg.pt`, SHA256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`;
- frozen batch: 32; Exp02.0 probe and one-epoch smoke PASS;
- Exp02.1: 100 epochs complete; best epoch 99; `best.pt` SHA256 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`;
- primary metric: independent VAL mask mAP50-95 `0.29898123412927496`;
- Exp02.2 val-only size/error audit PASS; test untouched.

## Gate decision

**Exp02 Baseline Gate = STOP pending user review.** Training and final evaluation are usable, but epoch 1--5 contain NaN in four val loss fields, violating the explicit `no NaN/Inf` PASS condition. Train losses contain no NaN/Inf, epoch 6--100 and final checkpoints are finite/loadable.

No Exp02.3, Exp03+, resolution ablation, threshold sweep, hard mining or alternate model has been started. The next action requires explicit user decision on whether early-val-loss NaN should trigger a controlled baseline rerun or be accepted as an Ultralytics cold-start validation artifact.
