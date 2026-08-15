# Exp10 Final Verify review

Exp10 controlled restart and unified three-seed VAL are complete. Final recommendation: **HARD_MINING_NOT_CONFIRMED**.

The seed43 controlled Treatment restart passed 30/30. Seed44 Baseline 100/100, hard pool 201/668, Control 30/30, and Treatment 30/30 all passed. Both seed44 arms used 20,040 samples and 331 optimizer steps. No run changed formal hyperparameters or silently downgraded AMP.

Treatment-Control Mask mAP50-95 deltas were `+0.030354`, `-0.006673`, and `-0.040311` for seeds 42/43/44. Positive count=1/3; mean=`-0.005543`, sample std=`0.035346`. Hard Mining therefore does not satisfy the robustness rule. The recommended Candidate Freeze choice is **Baseline**.

This is a recommendation only. Candidate Freeze has not been executed. Exp11 and TEST remain forbidden pending explicit user/ChatGPT review; `test_accessed=false`.

Evidence:

- `results/final_verify/exp10_three_seed_summary.csv`
- `results/final_verify/exp10_paired_deltas.csv`
- `results/final_verify/exp10_aggregate_summary.json`
- `results/final_verify/figures/exp10/`
- `results/final_verify/exp10_artifact_verification.json`
