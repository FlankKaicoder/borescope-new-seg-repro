# Exp13.2-A2 Batch03 user decision registration handoff

Registration UTC `2026-10-23T10:43:43.207444+00:00`. Gate `EXP13_2_A2_BATCH03_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`. This handoff describes the authorized registration and its checks; completed Git delivery evidence is in the external receipt below.

## Environment

Host `autodl-container-m6ahgmrgv8-57f0f00d`; repository `/root/autodl-tmp/borescope-new-seg-repro`; branch `exp13/vlm-semantic-audit`; origin `git@github-borescope:FlankKaicoder/borescope-new-seg-repro.git`. Starting local/live remote HEAD `cefae5ad9a7761153b0f7fe6fc7bf087e538101f`; tracked worktree/staging clean; starting index SHA256 `78698fd0422f809492b278c6ceb6b8bc61d302921a24d5fa1680c31236119da1`. Full pre-write snapshot: `/root/autodl-tmp/exp13_2_a2_batch03_registration_mi3uk2eb/starting_state_snapshot.json`.

## User decisions and scope

| variant_id | user_decision | decision_scope | user_comment |
|---|---|---|---|
| corrosion_9_v1 | REJECT | 当前审核候选 | 用户反馈效果很差，仍存在机械结构幻觉 |
| corrosion_9_v2 | REJECT | 当前审核候选 | 用户反馈效果很差，仍存在机械结构幻觉 |
| corrosion_6_v1 | ACCEPT_WITH_SCOPE_LIMITATION | 仅历史裁切视野 | 用户认为当前审核视野内效果可以接受；不代表裁切框外缺陷已修复 |
| corrosion_6_v2 | ACCEPT_WITH_SCOPE_LIMITATION | 仅历史裁切视野 | 用户认为当前审核视野内效果可以接受；不代表裁切框外缺陷已修复 |
| crack_469_v1 | REJECT | 当前审核候选 | 用户认为该版本仍有缺陷残留，不能接受 |
| crack_469_v2 | ACCEPT | 当前审核候选 | 用户认为该版本效果可以接受 |
| corrosion_1_v1 |  |  |  |
| corrosion_1_v2 |  |  |  |

8 unique candidates / 4 independent source groups, with six supplied decisions and two blank pending decisions. No preference transfers between variants. corrosion 6 accepts only the historical512×512 view:2789/4757=0.5862938826991801, outside1968pixels excluded. corrosion9 user feedback reports mechanical hallucinations; Crack469 v1 rejected/v2 accepted separately. corrosion1 decision/scope/comment all blank. These are user visual opinions, not independent industrial-expert certification, formal scores or semantic reassessment by Codex.

Canonical records: [decision report](exp13_2_a2_batch03_human_review_decision.md), [overlay](../results/exp13_restoration_feasibility/qa_20261023_batch03_user_review_overlay.csv), [validation](../results/exp13_restoration_feasibility/qa_20261023_batch03_registration_validation.json), unchanged [Batch03 manifest](../results/exp13_restoration_feasibility/batch03_20261023T095257Z_g3l3/batch03_manifest.csv). Overlay follows Batch02 columns with explicit extra provenance; scope clarification and missing bleed are documented in the decision report. All72formal quality cells blank; freeze/quality booleansfalse. Polygon-mask provenance remains annotation-derived; no pure self-supervision claim.

## Exact authorized file whitelist

New records:

- `results/exp13_restoration_feasibility/qa_20261023_batch03_user_review_overlay.csv`
- `results/exp13_restoration_feasibility/qa_20261023_batch03_registration_validation.json`
- `docs/exp13_2_a2_batch03_human_review_decision.md`
- `docs/exp13_2_a2_batch03_user_review_handoff.md`

Append-only records:

- `docs/PROJECT_STATE.md`
- `docs/DECISION_LOG.md`
- `docs/experiment_index.md`
- `results/experiment_registry.csv`

Decision D031 is new. Registry keeps11columns and appends one row; git_commit is starting evidence HEAD, not a self-referential final commit. No training code, dataset, cache, old manifest/template/overlay or image is staged.

## Verification and Git delivery

Validate8-key bijection, exact6decisions/two blanks, all provenance/hash/dimension/crop/technical fields, corrosion6scope,72blank cells,false booleans,CSV/JSON/Markdownlinks,append-only byte prefixes,registry11columns and whitelist. Before staging index stays unchanged. Check all256protected hashes,original tracked metadata except four append files,A0 88rows/792blank cells,Batch01/02 overlays/manifests and three pending decisions,and11,258legacy untracked path/size/mtime_ns/mode.

The committed validation JSON records completed pre-staging checks. Subsequent stage/commit/push outcomes must be read from `/root/autodl-tmp/exp13_2_a2_batch03_registration_mi3uk2eb/final_delivery_receipt.json`; it records exact final commit,live remote equality,clean tracked worktree,empty staging,index start/end hashes,file hashes,whitelist,diff checks and final historical protection. The final commit cannot contain its own SHA. Authorised staging/commit normally changes Git index; historical content is preserved. Final receipt is deliberately external to avoid an extra self-referencing commit. The final user-facing report will provide its local copy.

## Remaining questions and stop

corrosion1 v1/v2 await user decisions; Batch01's existing three pending decisions remain pending. ChatGPT audits this registration before any next task. User visual acceptance does not establish formal trusted-pair admission. No recomputation of cross-batch eligibility, dataset selection, model operation, images generated/modified, VAL/TEST images, paper access, training or next-stage execution occurred. Stop after commit/push and await ChatGPT audit.

```text
EXP13_2_A2_BATCH03_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```
