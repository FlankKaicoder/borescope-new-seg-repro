# Exp07 Stage2 fast repro

状态：**COMPLETE / NEGATIVE / VAL ONLY / test_accessed=false**

Stage1 固定 Exp02 baseline；仅评估 conf=0.05/0.10/0.15 与 Stage2 threshold=0.3/0.5/0.7。crop=1.2×、224×224；Stage2 只过滤/改类，mask 永远来自 Stage1。Mode A score 定义 `stage1_conf × p_defect`，Mode B 定义 `stage1_conf × classifier_pred_class_probability`。主结果采用 SUPCON_POSITIVE classifier；CE 结果保存在 `exp07_stage2_ce/`。

Stage1-only：conf .05 TP/FP/FN=170/628/126，.10=157/323/139，.15=144/195/152。Baseline .25 P/R/F1=0.5541/0.4155/0.4749，FP/FN=99/173。

最佳 Mode A：Stage1 .15、Stage2 .3，P/R/F1=0.5063/0.4088/0.4523，FP/FN=118/175。相对 baseline：Recall -0.0068、F1 -0.0226、FP +19、FN +2。

最佳 Mode B：Stage1 .15、Stage2 .7，P/R/F1=0.5729/0.3716/0.4508，FP/FN=82/186，wrong-class=4。相对 baseline：Recall -0.0439、F1 -0.0241、FP -17、FN +13。

91 个历史 low-conf recoverable GT：在选定 Stage1 conf=.15 下仅 21 个仍有 class-aware candidate，70 个在进入 Stage2 前已不可用；Mode A 保留 16、误滤 5；Mode B 保留 15、误滤 6。Mode A 删除/保留 FP=77/118（39.49%）；Mode B=113/82（57.95%）。Mode B correct/harmful/net reclassification=3/2/+1。

Latency（warmup 后 VAL）：Stage1 mean/median=1.792/1.792 ms/image；classifier=7.122/2.122 ms/candidate；end-to-end=27.950/8.157 ms/image。

AP=N/A：现有 fixed-point post-processing evaluator 无可靠 arbitrary-prediction COCO AP adapter，按 Fast Repro 规则不新写大型 AP 框架。Mode A/B 均未优于 YOLO conf=.25，Stage2=`NEGATIVE`；HardMining+FrozenStage2 probe=`NOT_RUN_BY_GATE`。
