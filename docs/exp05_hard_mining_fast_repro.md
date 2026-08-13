# Exp05 hard-mining fast repro

状态：**COMPLETE / POSITIVE_CANDIDATE / FAIR / test_accessed=false**

hard pool 只由 TRAIN 生成：201/668 张（30.09%）。Control/Treatment 均从同一 Exp02 best.pt 初始化，使用同一 replacement weighted sampler，每 epoch 668 samples，30 epochs。两臂均实际消费 20,040 images、331 optimizer updates；train loss 与 best/last 权重均 finite。Control hard draws=6077，Treatment=9244，唯一变量为 hard weight 1→2。

Control Mask P/R/mAP50/mAP50-95=0.72775/0.43759/0.51496/0.28096，FP/FN=104/179。Treatment=0.57919/0.55845/0.53577/0.31132，FP/FN=80/179。Treatment-Control：mAP50-95 +0.03035，Recall +0.12086，FP -24，FN 0。Crack/Burn/corrosion AP50-95 分别 +0.00530/+0.02500/+0.00668，Recall 分别 +0.08333/+0.16049/+0.03670。

best SHA256：Control `bfdd776ddeac3d5e062d3b9a2d124b5092544652450e650571b24335f47070a6`；Treatment `2d962735de9dd3f596780cc4e3bab6e745b06b5ae23aa4bc56f1f554c98f3d06`。证据：`results/fast_repro/exp05_hard_mining/`。
