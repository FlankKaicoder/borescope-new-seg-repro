# Final project summary

## Research chain

Raw images and professional polygon JSON → schema/pair/duplicate audit → JSON-to-YOLO segmentation conversion → near-duplicate group-aware split → frozen Dataset v1 → YOLO11n-seg baseline → size/error and confidence diagnostics → one-class and Hard Mining tests → ROI ResNet18 CE and SupCon → Stage2 evaluation → KD engineering Gate → SimSiam validity Gate → controlled three-seed verification → Hard Mining not confirmed → Baseline Candidate Freeze → one frozen TEST evaluation → project closure.

## Final result and interpretation

The selected seed44 Baseline achieved 0.325157 Mask mAP50-95 on frozen VAL and 0.271621 on the one-time TEST evaluation (delta -0.053536). TEST Mask P/R/mAP50 were 0.680727/0.498654/0.582704. At fixed confidence 0.25, F1 was 0.495127. Crack and Burn had the lowest Mask AP50-95; Tears, Burn and corrosion had the lowest recalls. No result was used to reopen model selection.

Hard Mining is not confirmed across seeds. SupCon is a positive ROI-representation result, Stage2 is negative, KD is unevaluated by engineering Gate, and SimSiam is unevaluated because its reconstruction did not update the trainable backbone. Resolution remains future work.

## Interview / Resume Evidence

Demonstrated polygon dataset engineering; JSON → YOLO segmentation; leakage-safe group-aware split; YOLO segmentation training and AMP numerical diagnosis; size/error taxonomy and threshold analysis; fair hard-sample mining with budget controls; ROI classification and SupCon representation learning; two-stage system evaluation; KD engineering probe; self-supervised validity audit; multi-seed robustness; disciplined one-time TEST access; reproducible Git registry and honest negative-result interpretation.

## Checkpoint preservation

最终权重：seed44 Baseline `best.pt`  
服务器绝对路径：`/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt`  
SHA256：`2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`  
冻结 VAL Mask mAP50-95：`0.3251567516`；单次 TEST Mask mAP50-95：`0.2716207089`。

权重文件未进入 Git。释放 AutoDL 实例前必须单独下载并校验 SHA256。
