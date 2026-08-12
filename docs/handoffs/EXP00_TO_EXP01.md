# EXP00 → EXP01 handoff

Generated: 2026-08-12 Asia/Shanghai  
Authoritative status: **Data Gate PASS FOR EXP01 / Environment Gate PASS / Exp01 allowed YES / Exp02 training NO**

> Policy closure (2026-08-12): this section supersedes the former pending state below. Professional JSON annotations are authoritative GT. The 24 unpaired images are `excluded_unpaired`. All 22 cross-label near-duplicate groups retain their professional annotations unchanged and remain indivisible split groups.

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

### Resolved by authoritative user policy

- 24 images without JSON: `excluded_unpaired`; do not treat as background and do not include in any split.
- 22 cross-label near-duplicate groups: retain all original professional annotations; no remediation or majority vote.
- Real acquisition IDs remain unavailable. Exp01 uses visual near-duplicate connected components as a documented leakage-control proxy.

## Git

- Branch: `main`
- Exp00.4/00.5 server execution baseline: `3d7a1c56930c0a6e93d8042cc883adafc80f1673`
- Final handoff/status documentation is committed after this baseline; use `git log -3 --oneline` and `git rev-parse HEAD` for the current HEAD.
- Expected working tree: clean except user-supplied `CODEX_TASK_NEW_BORESCOPE_SEG_REPRO.md` may remain untracked locally; server project should be clean.

## Gate

```text
Data Gate:
PASS FOR EXP01

Environment Gate:
PASS

Exp01 allowed:
YES

Formal model training:
NO
```

## Remaining limitation

真实视频、发动机、工件或采集批次 ID 仍不可用，不得编造；这不再阻塞本版 Exp01，但必须写入 provenance。

## Next

Proceed with Exp01.0--Exp01.2 and recompute the Dataset Freeze Gate. Do not start Exp02 or any formal YOLO training in this work cycle.
