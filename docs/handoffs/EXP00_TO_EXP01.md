# EXP00 → EXP01 handoff

Generated: 2026-08-12 Asia/Shanghai  
Authoritative status: **Data Gate STOP / Environment Gate PASS / Exp01 allowed NO**

## Dataset

| Item | Result |
|---|---|
| image_count | 993 |
| json_count | 969 |
| matched_count | 969 |
| image_without_json | 24 |
| actual classes | Burn, Crack, Dent, Material missing, Tears, Tip curl, corrosion |
| polygon issues | 0 program-detected geometry/schema issues |
| exact duplicates | 0 groups |
| near duplicates | 88 groups, 252 images, 265 similarity edges |
| near-duplicate label-set conflicts | 22 groups, 74 member images |
| raw manifest SHA256 | `1b0d6379800661c4b71e81db50f5b280bd46c5952e29f219c8f8427bc9c142a2` |

### Per-class counts

| Class | Image count | Instance count |
|---|---:|---:|
| Burn | 202 | 426 |
| Crack | 118 | 140 |
| Dent | 187 | 202 |
| Material missing | 216 | 249 |
| Tears | 66 | 66 |
| Tip curl | 35 | 35 |
| corrosion | 261 | 729 |

### Exp00.4 unpaired review

- 24/24 images were decoded and packaged with SHA256, dimensions and nearest labeled-image evidence.
- None met the frozen moderate/high visual-similarity thresholds; program hints are 24×`unknown`.
- Non-authoritative visual triage: 6 candidate_background, 11 possible_missing_annotation, 7 unknown, 0 evidence-backed possible_missing_json.
- No empty JSON was created and no image was assigned a final semantic status.
- Human review CSV: `results/dataset_audit/exp00_4_unpaired_review/artifacts/image_without_json.csv`.

### Exp00.5 conflict review

- 22 label-set conflict groups, 74 members, 22 original+GT-overlay sheets.
- Conflict composition: Dent↔Material missing 14 groups; Dent↔Tears 3; five other patterns 1 group each.
- Main observed risks: same/near-same edge defect named differently; Tip curl/Crack/Tears boundary instability; multi-instance incomplete annotation; some visually similar scenes may contain genuinely different defect locations.
- No label or JSON was changed. All 22 group decisions remain pending.
- Review CSV: `results/dataset_audit/exp00_5_conflict_review/artifacts/conflict_group_review.csv`.

## Environment

| Item | Result |
|---|---|
| GPU visibility | 1 × NVIDIA GeForce RTX 2080 Ti |
| GPU memory | nvidia-smi 22528 MiB; PyTorch 23069917184 bytes (~22GB) |
| Initial two-GPU plan | Superseded by Exp00 measured environment |
| Python | 3.12.3 |
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| torch CUDA runtime | 12.8; CUDA tensor smoke PASS |
| NVIDIA driver | 595.71.05; nvidia-smi reports CUDA capability 13.2 |
| training venv | `/root/autodl-tmp/borescope-new-seg-repro/.venv` |
| Ultralytics | 8.4.117; local `yolo11n-seg.yaml` model-load smoke PASS |
| OpenCV | opencv-python-headless 5.0.0; real image decode smoke PASS |
| pandas / sklearn / shapely | 3.0.5 / 1.9.0 / 2.1.2 |

The environment is ready for later smoke experiments, but **formal training remains prohibited by the Data Gate**. The old audit environment is preserved in `environment_report.md`; the isolated environment is in `training_environment_report.md`.

## Decisions

### Frozen

- Original data is read-only.
- Instance segmentation is the primary task.
- Historical 537-image work is a method-idea source, not bit-exact reproduction.
- GAN, CutPaste, DRAEM, medical-polyp transfer and old 9-class P2/P2+ECA are excluded.
- SimSiam is a later extension.
- Current single 22GB GPU topology supersedes the initial two-GPU assumption.
- Exact/high-confidence near duplicates must remain in one future split group.
- No final split exists; test discipline remains mandatory.

### Pending human confirmation

- Final semantics of all 24 images without JSON.
- Correct label/annotation action for each of 22 conflict groups.
- Whether unresolved items are repaired or excluded to a review pool.
- Whether real video/engine/workpiece/acquisition IDs can be provided.

## Git

- Branch: `main`
- Exp00.4/00.5 server execution baseline: `3d7a1c56930c0a6e93d8042cc883adafc80f1673`
- Final handoff/status documentation is committed after this baseline; use `git log -3 --oneline` and `git rev-parse HEAD` for the current HEAD.
- Expected working tree: clean except user-supplied `CODEX_TASK_NEW_BORESCOPE_SEG_REPRO.md` may remain untracked locally; server project should be clean.

## Gate

```text
Data Gate:
STOP

Environment Gate:
PASS

Exp01 allowed:
NO
```

## Minimum questions requiring user/domain-expert answers

1. 对 24 张无 JSON 图片，逐张确认：background / missing annotation / missing JSON / exclude-unknown？
2. 对 22 个冲突组，确认正确类别、是否漏标实例，以及修复还是整组排除？
3. 是否能提供视频、发动机、工件或采集批次 ID 作为真实 group boundary？

## Next

**DO NOT START EXP01.**

收到人工决定后，先生成修复/排除 proposal 和审查后 manifest，再重新计算 Data Gate。即使所有 Gate PASS，也必须先提交 Exp01.0–01.2 具体计划并等待用户确认，不能自动转换数据、冻结 split 或训练 YOLO。
