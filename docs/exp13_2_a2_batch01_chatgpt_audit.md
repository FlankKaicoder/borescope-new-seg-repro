# Exp13.2-A2 Batch 01 ChatGPT audit archival

Recorded 2026-10-23 UTC. Evidence source: the user's explicit Batch 02 task statement relaying ChatGPT's completed Batch 01 audit. This task did not repeat the visual review or obtain independent industrial-expert annotations.

## Audit conclusion and scope

`PASS / BOUNDED` applies to Batch 01 user-decision registration and spatial-consistency audit, covering 8 historical candidates from 4 independent sources. It does not certify restoration quality, full-RAW repair, a preferred variant, or a trusted training pair. Historical Gate entries remain time-stamped snapshots.

The [original decision overlay](../results/exp13_restoration_feasibility/qa_20261023_batch01_user_review_overlay.csv), [decision report](exp13_2_a2_batch01_human_review_decision.md), and [spatial audit](exp13_2_a2_batch01_spatial_audit.md) remain unchanged. Three decisions remain `PENDING_USER_DECISION`: `crack_111_v1`, `crack_146_v1`, `crack_146_v2`. Batch 02 does not adjudicate these candidates.

| Source | Registration-time user decision | Subsequent technical evidence | Bounded interpretation |
|---|---|---|---|
| Crack 470 v1/v2 | `USER_ACCEPTED_VISUAL_PENDING_SPATIAL_AUDIT` | 3,134 / 3,134 original nonzero mask pixels lie inside the historical crop; fraction 1.0; spatial audit completed | The historical overlay label preserves the state when registered. The current coordinate result is separately known. The conflicting historical textual crop warning is retained, not silently erased. Neither variant is ranked or frozen. |
| corrosion 13 v1/v2 | `USER_ACCEPTED_VISUAL_CROP_SCOPED` | 2,747 / 6,225 original mask pixels retained; 0.441285140562249; 3,478 outside crop | Visual acceptance applies only to the reviewed historical 512x512 field. Cropped-away corrosion cannot be called repaired; no full-RAW acceptance or removal of source-level crop risk. |

Crack 111 v2 retains the user's visual acceptance. For Crack 146, the user's explanation that green markings and blur came from the input remains authoritative; those observations cannot directly establish generated artifacts. There is no new decision on either Crack 146 variant.

## Original manifest preservation

The historical file `/root/autodl-tmp/exp13_2_a2_batch01_visual_review_20261023T062216Z_078cbbde/batch01_manifest.csv` was present and matched the expected SHA256 exactly:

```text
b58691a5b8745c22bf2c303df38d14cdd5bde26d505c039d598416d802ff10a2
```

Its original bytes were copied to the [archived manifest](../results/exp13_restoration_feasibility/qa_20261023_batch01_manifest.csv). Original and archive hashes agree; nothing was regenerated or edited. Original A0 tables and Batch 01 overlay remain independent historical records.

## Subsequent permitted work

Batch 02 prepares existing TRAIN-only R1 assets for another 4 sources / 8 candidates. Its schema separates `user_decision` from `technical_audit_status`. Technical file/coordinate consistency is not a user visual decision. See the [Batch 02 report](../results/exp13_restoration_feasibility/qa_20261023_batch02_review_report.md).

Current handoff Gate: `EXP13_2_A2_BATCH02_REVIEW_PACKET_PREPARED_AWAITING_CHATGPT_AUDIT`.

```text
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

No new defect-quality visual review, model invocation, generation, training, inference, evaluation, VAL/TEST image access, or pair freezing occurred. Stop for ChatGPT audit and user visual review.
