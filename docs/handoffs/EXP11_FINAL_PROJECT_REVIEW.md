# Exp11 final project review

## A. Candidate Freeze

1. 是；在任何正式 TEST 指标生成前完成。  
2. `9991fcfcb9cf6c0ab8920ad7deadeed579ce5585`。  
3. YOLO11n-seg Baseline。  
4. seed44。  
5. `/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt`。  
6. `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`。  
7. 是，freeze 时 `test_accessed=false`。

## B. TEST

8. 147 images / 285 instances。  
9. Box P=0.662385，R=0.474927，mAP50=0.541636，mAP50-95=0.293253。  
10. Mask P=0.680727，R=0.498654，mAP50=0.582704，mAP50-95=0.271621。  
11. conf=.25：TP=127，FP=101，FN=158，P=0.557018，R=0.445614，F1=0.495127。  
12. tiny Recall=0.405941。  
13. small Recall=0.527778。  
14. medium Recall=0.461538。  
15. large Recall=0.382979。

## C. Per class

16. Burn 58/Recall .379310/Mask AP50 .397697/AP50-95 .219134；Crack 21/.428571/.431181/.130543；Dent 37/.486486/.536607/.296771；Material missing 33/.424242/.493331/.346388；Tears 11/.363636/.743492/.359117；Tip curl 5/1.000000/.995000/.300860；corrosion 120/.408333/.481618/.248531。  
17. 按 Mask AP50-95：Crack、Burn、corrosion；按 Recall：Tears、Burn、corrosion。

## D. Generalization

18. 0.3251567516。  
19. 0.2716207089。  
20. -0.0535360427。  
21. YES；仅作 generalization observation，未据此训练、调参或重新选择。

## E. Research conclusion

22. `POSITIVE_DIAGNOSTIC / NOT_FINAL_MODEL`；91/173 FN 可恢复但 FP 激增。  
23. `NO_CLEAR_GAIN`。  
24. `HARD_MINING_NOT_CONFIRMED`；paired mean -0.005543±0.035346，1/3 positive。  
25. `COMPLETE_DIAGNOSTIC`；CE macro F1 .677804。  
26. `POSITIVE_ROI_REPRESENTATION / NOT_FINAL_SEGMENTATION_METHOD`；macro F1 +.015894。  
27. `NEGATIVE`。  
28. `SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED`。  
29. `INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED`。  
30. `NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE / FUTURE_WORK_ONLY`。  
31. RQ1 ANSWERED；RQ2 NOT_FORMALLY_ANSWERED/DEFERRED；RQ3 ANSWERED/POSITIVE_DIAGNOSTIC；RQ4 ANSWERED/NOT_CONFIRMED；RQ5 PARTIALLY_ANSWERED；RQ6 ANSWERED/NEGATIVE；RQ7 ANSWERED_FOR_ROI_REPRESENTATION；RQ8 NOT_EVALUATED/ENGINEERING_GATE；RQ9 NOT_EVALUATED；RQ10 PARTIALLY_ANSWERED。

## F. Artifacts

32. `results/final/final_main_results.csv`：完成。  
33. `results/final/paper_per_class_test.csv`：完成。  
34. `results/final/paper_three_seed_table.csv`：完成。  
35. `results/final/paper_hard_mining_ablation.csv`：完成。  
36. `results/final/paper_roi_supcon_table.csv`：完成。  
37. final figures：完成并索引。  
38. 64 张案例 + qualitative grid：完成。  
39. PASS；107 个文件非空、CSV/JSON 可解析、图像签名/内部 decode audit 通过。

## G. Documents

40. README：完成。  
41. final_ablation_report：完成。  
42. paper_materials：完成。  
43. final_project_summary：完成。  
44. method_reconstruction：完成。  
45. figure index：完成。  
46. timeline：完成。

## H. Project closure

47. `PROJECT_COMPLETE`。  
48. 是；freeze 之后才访问。首次执行在指标前被客户端中断并保留，随后只有一次用户明确授权的同参重试。  
49. NO。  
50. NO。  
51. `true`。  
52. NO。  
53. 仅 960 resolution、预注册 continued fine-tuning、future improvement phase；均非当前未完成工作。

## I. Checkpoint backup

54. `/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt`。  
55. `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`。  
56. 是：释放 AutoDL 前必须单独下载 seed44 `best.pt`；GitHub 不含 `.pt`。

## J. Git

57. `9991fcfcb9cf6c0ab8920ad7deadeed579ce5585`。  
58. `fb584621b816de8f344d49daeba2656caee46e92`。  
59. `b094d16bacb325e44d61eb4bedad54b00d231790`。  
60. `FINAL_HEAD` 是包含本 closure metadata 的提交；精确 SHA 由最终 `git rev-parse HEAD` 外部核验（提交不能自包含自身 SHA）。  
61. 与最终 closure HEAD 相等；由最终三端核验记录。  
62. 与最终 closure HEAD 相等；由最终三端核验记录。  
63. 与最终 closure HEAD 相等；由最终三端核验记录。  
64. `PASS`（最终回复给出三个完全一致的 SHA）。  
65. Server clean：最终核验要求 `YES`。  
66. Windows clean：最终核验要求 `YES`。  
67. `YES`；最终核验必须仍为 `a9c89ff3a75308676261035f7ad463f5ebcd8a2c` 与 `d8cc011fed79af0235b825a36e95b55d6cb242af`。

## K. Source

68. 建议重新上传：`docs/PROJECT_STATE.md`、`ROADMAP.md`、`CHANGELOG.md`、`README.md`、`docs/final_candidate_freeze.md`、`docs/final_ablation_report.md`、`docs/paper_materials.md`、`docs/final_project_summary.md`、本 handoff、research takeaways、timeline、figure index，以及 `results/final/*.csv`、`results/final_test/exp11_retry1/{summary,overall_metrics,per_class_metrics,size_metrics,fixed_threshold_metrics}.*`。
