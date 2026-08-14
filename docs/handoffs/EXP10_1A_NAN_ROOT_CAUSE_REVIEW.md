# Exp10.1a NaN root-cause review

Exp10.1a is complete and waiting for review. The gate result is **CASE C — FP32_MODEL_OR_OPTIMIZATION_INSTABILITY**.

The first non-finite event is epoch1 batch1, before optimizer step1. Input, targets, sampler mapping, TRAIN data, pre-step parameters, and optimizer state are valid/finite; raw model output becomes non-finite in forward. Independent exact replay of the saved augmented batch and pre-step state is non-finite under both AMP and FP32. The same result holds at the baseline initial state because it is tensor-identical to the pre-step state. FP32 C2PSA qk/softmax remain finite, so this is not the prior FP16-only qk overflow.

The Treatment retry did experience an unexpected runtime AMP self-check downgrade (`amp=true` requested, effective AMP=false), but this is secondary: exact AMP replay also fails. No repair, hyperparameter change, formal rerun, seed44, VAL, TEST, Candidate Freeze, or Exp11 was performed.

Current status: `EXP10_DIAGNOSTIC_COMPLETE_WAITING_REVIEW`. Exp10 is still stopped/incomplete. `val_accessed_for_probe=false`; `test_accessed=false`.
