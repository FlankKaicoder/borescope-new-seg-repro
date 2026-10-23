# Exp13.2-A2 Batch 02 user decision registration handoff

Registration UTC: `2026-10-23T09:14:43.048190+00:00`. Gate: `EXP13_2_A2_BATCH02_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`.

Host `autodl-container-m6ahgmrgv8-57f0f00d`; repository `/root/autodl-tmp/borescope-new-seg-repro`; branch `exp13/vlm-semantic-audit`; origin `git@github-borescope:FlankKaicoder/borescope-new-seg-repro.git`. Starting local/live remote HEAD `a5f1038c72e6e914450ec6a6f1929337169172a8`. Starting index SHA256 `b2ab5c0e8bb8381813d33ba0a766a81709481baadecad8c6c940d3661c8482d5`. The final commit cannot contain its own hash; resolve the commit containing this handoff and consult the external final receipt below for exact local/remote final HEAD and commit/push results.

## Decisions and evidence

| variant_id | user_decision | decision_scope | user_comment |
|---|---|---|---|
| crack_86_v1 | ACCEPT | 当前审核候选 | 用户接受 |
| crack_86_v2 | ACCEPT | 当前审核候选 | 用户接受 |
| crack_260_v1 | ACCEPT | 当前审核候选 | 用户接受 |
| crack_260_v2 | ACCEPT | 当前审核候选 | 用户接受 |
| corrosion_2_v1 | ACCEPT_WITH_SCOPE_LIMITATION | 仅历史裁切视野 | 用户接受当前审核视野；不得声称裁切框外缺陷已修复 |
| corrosion_2_v2 | ACCEPT_WITH_SCOPE_LIMITATION | 仅历史裁切视野 | 用户接受当前审核视野；不得声称裁切框外缺陷已修复 |
| corrosion_8_v1 | REJECT | 当前审核候选 | 用户认为该版本效果不理想 |
| corrosion_8_v2 | ACCEPT_WITH_RESERVATION | 当前审核候选 | 用户认为该版本基本可以接受，保留意见 |

[Decision report](exp13_2_a2_batch02_human_review_decision.md), [independent overlay](../results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_overlay.csv), [validation](../results/exp13_restoration_feasibility/qa_20261023_batch02_registration_validation.json) and unchanged [Batch 02 manifest](../results/exp13_restoration_feasibility/qa_20261023_batch02_manifest.csv) are canonical evidence. All 8 variants join uniquely to 4 independent source groups. User decisions and scopes are registered faithfully in meaning. Some comments are normalized summaries and must not be quoted as verbatim user words. Original decision provenance, registration UTC, decision enums and scopes remain unchanged. Technical status is separate. The scope/reservation decisions do not certify full RAW or formal restoration quality.

corrosion 2 mask retention remains 1314/2394 = 0.5488721804511278; 1080 annotation pixels outside the historical crop remain outside acceptance. corrosion 8 v1 is REJECT while v2 is ACCEPT_WITH_RESERVATION. Seven decisions containing acceptance are not seven trusted pairs; variants are grouped by source.

## File whitelist and delivery checks

New files: `results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_overlay.csv`, `results/exp13_restoration_feasibility/qa_20261023_batch02_registration_validation.json`, `docs/exp13_2_a2_batch02_human_review_decision.md`, `docs/exp13_2_a2_batch02_user_review_handoff.md`.

Append-only files: `docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md` (D028), `docs/experiment_index.md`, `results/experiment_registry.csv`. Registry retains its 11 columns; git_commit records the starting evidence HEAD, not a self-referential final SHA. No other files are authorized.

Before staging, verify the exact decision table, eight unique candidate keys, all original manifest/provenance values, blank formal quality fields, all false freeze/quality booleans, all protected hashes, unchanged A0 792 blanks, unchanged Batch 01 manifest/overlay/three pending decisions, and all 11,258 legacy untracked paths with size/mtime_ns/mode. Verify append-only byte prefixes and registry structure, Markdown links, `git diff --check`, index unchanged until authorized explicit staging, and exact cached whitelist/content. No `git add .`, cleanup or destructive Git operations.

External complete starting snapshot: `/root/autodl-tmp/exp13_2_a2_batch02_user_registration_51vexfc8/starting_state_snapshot.json`. Pre-staging/staged/commit/final receipts are stored in the same unique directory. Final receipt: `/root/autodl-tmp/exp13_2_a2_batch02_user_registration_51vexfc8/final_delivery_receipt.json`; it records final HEAD, live remote equality, clean tracked worktree/index, historical hashes, legacy metadata and diff checks after push. Validation JSON in this commit describes pre-staging evidence; the external final receipt describes completed Git delivery.

## Unresolved matters and stop

ChatGPT audit and formal pair-admission/quality judgment remain outstanding. No new visual assessment, expert opinion, quantitative pass rate or training-quality conclusion is made. Batch 01's three pending user decisions remain pending. No Batch 03, model execution, image generation, training, SSL, Exp13.2-B, pair freeze, VAL/TEST images or paper repository access. Stop after commit/push and await ChatGPT audit.

```text
EXP13_2_A2_BATCH02_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```
