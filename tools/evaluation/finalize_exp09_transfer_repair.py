#!/usr/bin/env python3
"""Finalize documentation and evidence for the Exp09.2a real transfer failure."""
from pathlib import Path
import csv
import json

import cv2
import matplotlib.pyplot as plt

ROOT = Path("/root/autodl-tmp/borescope-new-seg-repro")
FAST = ROOT / "results/fast_repro"
REPAIR = FAST / "exp09_transfer_repair"
REPORT = json.loads((REPAIR / "transfer_gate_report.json").read_text(encoding="utf-8"))
assert REPORT["status"] == "REAL_TRANSFER_FAILURE"
assert REPORT["test_accessed"] is False


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


figure_dir = FAST / "figures/exp09_simsiam"
figure_dir.mkdir(parents=True, exist_ok=True)
figure = figure_dir / "exp09_transfer_verification_summary.png"
labels = ["Immediate\nparameters", "FP32\nround-trip", "Native FP16\nexplained", "Changed vs COCO\nparameters"]
values = [120, 240, 240, 0]
denominators = [120, 240, 240, 120]
colors = ["#2f855a", "#2f855a", "#2f855a", "#c53030"]
plt.figure(figsize=(8, 4.5))
bars = plt.bar(labels, values, color=colors)
for bar, value, denominator in zip(bars, values, denominators):
    plt.text(bar.get_x() + bar.get_width() / 2, max(value, 3) + 4, f"{value}/{denominator}", ha="center")
plt.ylim(0, 275)
plt.ylabel("Verified tensor count")
plt.title("Exp09.2a revised transfer verification")
plt.tight_layout()
plt.savefig(figure, dpi=180)
plt.close()
image = cv2.imread(str(figure))
assert image is not None and image.size > 0

master_path = FAST / "fast_repro_master_summary.csv"
master = list(csv.DictReader(master_path.open(encoding="utf-8-sig")))
for row in master:
    if row["Experiment"] == "Exp09 SimSiam":
        row["Conclusion"] = "REAL_TRANSFER_FAILURE"
        row["Value"] = "N/A"
write_csv(master_path, master)

summary_path = FAST / "fasttrack_c_summary.csv"
summary = list(csv.DictReader(summary_path.open(encoding="utf-8-sig")))
for row in summary:
    if row["Experiment"] == "Exp09 downstream":
        row.update({"Formal_training": "NO", "VAL": "NO", "Status": "NOT_RUN_BY_GATE", "Test_accessed": "false", "Evidence": "exp09_transfer_repair/transfer_gate_report.json"})
summary.append({"Experiment": "Exp09.2a transfer repair", "Formal_training": "NO", "VAL": "N/A", "Status": "REAL_TRANSFER_FAILURE", "Test_accessed": "false", "Evidence": "exp09_transfer_repair/transfer_gate_report.json"})
write_csv(summary_path, summary)

matrix_path = FAST / "method_status_matrix.csv"
matrix = list(csv.DictReader(matrix_path.open(encoding="utf-8-sig")))
for row in matrix:
    if row["Method"] == "SimSiam":
        row.update({"Result": "REAL_TRANSFER_FAILURE", "Primary_evidence": "0/120 trainable parameters changed vs COCO", "Keep_for_final_verify": "NO", "Reason": "Frozen SSL export changed only BN buffers"})
write_csv(matrix_path, matrix)

registry_path = ROOT / "results/experiment_registry.csv"
registry = list(csv.DictReader(registry_path.open(encoding="utf-8-sig")))
fields = list(registry[0])
registry = [row for row in registry if row["experiment_id"] != "Exp09.2a"]
registry.append(dict(zip(fields, [
    "Exp09.2a", "SimSiam backbone transfer verification repair", "REAL_TRANSFER_FAILURE",
    "2026-08-13T11:00:00Z", "2026-08-13T11:10:00Z", "06c55ba",
    "/root/autodl-tmp/borescope-new-seg-data/v1", "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d",
    "bash scripts/exp09_transfer_repair.sh", "results/fast_repro/exp09_transfer_repair",
    "Old 80 mismatches were BN running stats; immediate and round-trip audits pass, but 0/120 trainable parameters changed vs COCO; downstream not run; test untouched",
])))
write_csv(registry_path, registry, fields)

manifest_path = FAST / "artifact_manifest.csv"
manifest = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
fields = list(manifest[0])
rel = str(figure.relative_to(ROOT))
manifest = [row for row in manifest if row["file_path"] != rel]
manifest.append({"experiment": "exp09_transfer_repair", "artifact_type": "png", "file_path": rel, "file_size": str(figure.stat().st_size), "decode_pass": "True"})
write_csv(manifest_path, manifest, fields)

(ROOT / "docs/exp09_transfer_verification_repair.md").write_text("""# Exp09.2a SimSiam backbone transfer verification repair

Final gate: **REAL_TRANSFER_FAILURE**. `test_accessed=false`. Exp09 downstream training and VAL are `NOT_RUN_BY_GATE`.

## Why the old gate could misclassify

Ultralytics 8.4.117 was inspected directly from the project environment. `Model.save()` calls `deepcopy(self.model).half()` at `engine/model.py:364`; `Trainer.save_model()` serializes a half-precision EMA at `engine/trainer.py:725`; `strip_optimizer()` calls `x["model"].half()` at `utils/torch_utils.py:828`. Raw FP32 byte hashes therefore cannot be the sole native-checkpoint criterion. Site-packages were not modified.

The old 80 byte mismatches were exactly 40 BN `running_mean` and 40 BN `running_var` buffers. None were trainable parameters. Native round-trip was 240/240 exact after reproducing FP16 quantization; FP32 state-dict control was 240/240 exact.

## Revised transfer evidence

- Key/shape gate: expected=240, loaded=240, missing=0, unexpected=0, shape mismatch=0.
- Immediate in-memory load before any forward: all 120/120 trainable parameters exactly matched the SimSiam export after dtype normalization.
- Eval forward changed zero BN buffers.
- Diagnostic train-mode forward changed all 40 running means, 40 running variances, and 40 batch counters; this copy never contaminated formal initialization.
- The decisive failure: compared with the official COCO initialization, 0/120 trainable backbone parameters had changed. All 120 changed tensors were BN buffers. There is therefore no trainable parameter satisfying `COCO != SimSiam == downstream`.

The revised verifier proves that loading and checkpoint serialization work, but it also proves the frozen SSL export contains no learned trainable backbone update. Under the explicit Gate definition this is a real transfer failure, not a serialization false alarm. SSL was not rerun or tuned, and downstream training was forbidden.
""", encoding="utf-8")

exp09_path = ROOT / "docs/exp09_simsiam_fast_repro.md"
exp09_path.write_text("""# Exp09 SimSiam fast reproduction

Final status: **REAL_TRANSFER_FAILURE**. `test_accessed=false`.

The frozen SSL run remains valid as an execution result: 668 TRAIN images only, 100/100 epochs, batch 32, finite loss/checkpoint, and no representation-collapse signal. It was not rerun in Exp09.2a.

The original native-checkpoint byte gate was over-sensitive because Ultralytics 8.4.117 saves models in FP16. Revised verification established expected/loaded=240/240, missing/unexpected/shape mismatch=0, immediate trainable transfer=120/120, FP32 round-trip=240/240, and native FP16-explained round-trip=240/240. The old 80 mismatches were exactly BN running-mean/running-var buffers.

However, comparison with official COCO initialization found that the frozen SSL export changed 0/120 trainable backbone parameters and changed only 120 BN buffers. It therefore cannot prove learned SimSiam weights were transferred. The revised gate is `REAL_TRANSFER_FAILURE`; downstream 100-epoch training and VAL comparison are `NOT_RUN_BY_GATE`. See `docs/exp09_transfer_verification_repair.md`.
""", encoding="utf-8")

(ROOT / "docs/handoffs/FASTTRACK_C_REVIEW.md").write_text("""# FastTrack-C final review

FastTrack-C is closed with **REAL_TRANSFER_FAILURE**, not complete success. Exp08 remains `SKIPPED_BY_ENGINEERING_GATE`. Exp09 SSL completed without collapse, but Exp09.2a proved its frozen export changed 0/120 trainable backbone parameters relative to COCO; only BN buffers changed. Consequently Exp09 downstream and VAL are `NOT_RUN_BY_GATE`. `test_accessed=false` throughout.

The highest completed formal segmentation result remains Exp05 Hard Mining (VAL mask mAP50-95 0.311318 versus baseline 0.298981, delta +0.012337). Old-method Final Verify candidate: Exp05 Hard Mining. Extension candidate: NONE.

Recommendation only: after explicit user authorization, Exp10 may evaluate Baseline and Exp05 Hard Mining across three seeds, then freeze the candidate before any one-time Exp11 TEST. This review does not authorize or execute either phase.
""", encoding="utf-8")

state_path = ROOT / "docs/PROJECT_STATE.md"
state = state_path.read_text(encoding="utf-8")
state = state.replace("FastTrack-C stopped at Exp09 backbone-transfer Hard Gate", "FastTrack-C closed at Exp09.2a REAL_TRANSFER_FAILURE")
state = state.replace("Exp09 SSL PASS/no collapse; downstream blocked by transfer verification", "Exp09.2a REAL_TRANSFER_FAILURE; downstream NOT_RUN_BY_GATE")
state = state.replace("Case C confirmed: epoch 1--5 training-validation AMP C2PSA qk matmul overflows before loss; explicit no-NaN Baseline Gate condition remains violated. Real acquisition IDs also remain unavailable.", "Exp09 frozen SSL export changed 0/120 trainable backbone parameters versus COCO; extension candidate is NONE. Real acquisition IDs remain unavailable. Exp02 AMP overflow is a documented limitation covered by PASS_WITH_NUMERICAL_WAIVER, not an active blocker.")
state = state.replace("Exp09 BACKBONE_TRANSFER_HARD_GATE; FastTrack-C stopped", "Exp09.2a REAL_TRANSFER_FAILURE; FastTrack-C closed")
state = state.replace("None; review Exp09 transfer Hard Gate before any further experiment", "None automatically; Exp10 requires explicit user authorization")
state = state.replace("None; FastTrack-C is stopped and no further formal training is authorized", "None; Exp09 downstream is forbidden by gate and Exp10/Exp11 are not authorized")
state = state.replace("2026-08-13T18:45:00+08:00", "2026-08-13T19:20:00+08:00")
state = state.replace("Wait for user review of `docs/handoffs/FASTTRACK_C_REVIEW.md`. Do not start Exp10/Exp11 or access test automatically.", "Wait for explicit user authorization before Exp10. Do not start Exp10/Exp11 or access TEST automatically.")
state_path.write_text(state, encoding="utf-8")

roadmap_path = ROOT / "ROADMAP.md"
roadmap = roadmap_path.read_text(encoding="utf-8")
roadmap = roadmap.replace("- [ ] Exp09 downstream: STOP_BY_BACKBONE_TRANSFER_HARD_GATE", "- [x] Exp09.2a transfer repair: REAL_TRANSFER_FAILURE\n- [x] Exp09 downstream/VAL: NOT_RUN_BY_GATE")
roadmap_path.write_text(roadmap, encoding="utf-8")

index_path = ROOT / "docs/experiment_index.md"
index = index_path.read_text(encoding="utf-8")
old = "| Exp09 | SimSiam YOLO backbone adaptation | STOP / TRANSFER_HARD_GATE | `docs/exp09_simsiam_fast_repro.md` |"
new = "| Exp09/09.2a | SimSiam adaptation and transfer repair | REAL_TRANSFER_FAILURE / downstream NOT_RUN_BY_GATE | `docs/exp09_transfer_verification_repair.md` |"
index = index.replace(old, new)
index_path.write_text(index.rstrip() + "\n", encoding="utf-8")

changelog_path = ROOT / "CHANGELOG.md"
changelog = changelog_path.read_text(encoding="utf-8")
entry = "- Exp09.2a corrected the byte-hash verification: old mismatches were 80 BN running-stat buffers and native FP16 round-trip is explainable, but the frozen SSL export changed 0/120 trainable parameters versus COCO. Final gate: REAL_TRANSFER_FAILURE; downstream/VAL NOT_RUN_BY_GATE.\n"
if entry not in changelog:
    changelog = changelog.replace("## Unreleased\n", "## Unreleased\n\n" + entry)
changelog_path.write_text(changelog, encoding="utf-8")

print(json.dumps({"status": REPORT["status"], "test_accessed": False, "artifact_count": len(manifest), "decode_pass": True}, indent=2))
