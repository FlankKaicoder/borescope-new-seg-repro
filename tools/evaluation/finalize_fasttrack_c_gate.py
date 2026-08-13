#!/usr/bin/env python3
"""Finalize FastTrack-C evidence after the Exp09 transfer hard gate."""
from pathlib import Path
import csv
import hashlib
import json

import cv2
import matplotlib.pyplot as plt

ROOT = Path("/root/autodl-tmp/borescope-new-seg-repro")
FAST = ROOT / "results/fast_repro"
EXP08 = FAST / "exp08_kd"
EXP09 = FAST / "exp09_simsiam"
FIG = FAST / "figures/exp09_simsiam"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def append_unique_csv(path, rows, key):
    old = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    replacements = {row[key]: row for row in rows}
    merged = [row for row in old if row[key] not in replacements]
    merged.extend(rows)
    write_csv(path, merged, list(old[0]))


ssl = load_json(EXP09 / "ssl/summary.json")
transfer = load_json(EXP09 / "transfer_report.json")
history = list(csv.DictReader((EXP09 / "ssl/history.csv").open(encoding="utf-8-sig")))
FIG.mkdir(parents=True, exist_ok=True)
epochs = [int(row["epoch"]) for row in history]

plt.figure()
plt.plot(epochs, [float(row["loss"]) for row in history])
plt.xlabel("epoch")
plt.ylabel("SimSiam loss")
plt.tight_layout()
plt.savefig(FIG / "exp09_simsiam_loss.png", dpi=160)
plt.close()

plt.figure()
plt.plot(epochs, [float(row["feature_std"]) for row in history], label="feature std")
plt.plot(epochs, [float(row["embedding_std"]) for row in history], label="embedding std")
plt.xlabel("epoch")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "exp09_feature_std.png", dpi=160)
plt.close()

plt.figure(figsize=(7, 4))
plt.bar(
    ["expected", "reload exact", "changed"],
    [transfer["expected_tensor_count"], transfer["loaded_tensor_count"], transfer["changed_tensor_count"]],
)
plt.ylabel("tensor count")
plt.title("Exp09 backbone transfer verification")
plt.tight_layout()
plt.savefig(FIG / "exp09_transfer_verification.png", dpi=160)
plt.close()

exp08_summary = {
    "status": "SKIPPED_BY_ENGINEERING_GATE",
    "test_accessed": False,
    "teacher_sha256": "8e22f17c029eb0f3cb9416673a3503e0d37c7b98b91f126da2107e23fe58c32b",
    "student_feature_layer": 4,
    "student_feature_module": "C3k2",
    "student_feature_stride": 8,
    "student_feature_channels": 128,
    "roi_align_output": [7, 7],
    "teacher_crop_scale": 1.2,
    "teacher_crop_size": [224, 224],
    "formal_aux_ce_run": False,
    "formal_kd_run": False,
    "val_run": False,
    "engineering_gate_reason": "Online teacher ROI extraction exceeded 70 seconds for one batch=32 training step after the OOM-safe chunking fix.",
}
EXP08.mkdir(parents=True, exist_ok=True)
(EXP08 / "summary.json").write_text(json.dumps(exp08_summary, indent=2) + "\n", encoding="utf-8")

append_unique_csv(
    FAST / "fast_repro_master_summary.csv",
    [
        {"Experiment": "Exp08 AUX_CE", "Metric_domain": "Segmentation", "Primary_metric": "Mask mAP50-95", "Value": "N/A", "Conclusion": "SKIPPED_BY_ENGINEERING_GATE"},
        {"Experiment": "Exp08 KD", "Metric_domain": "Segmentation", "Primary_metric": "Mask mAP50-95", "Value": "N/A", "Conclusion": "SKIPPED_BY_ENGINEERING_GATE"},
        {"Experiment": "Exp09 SimSiam", "Metric_domain": "Segmentation", "Primary_metric": "Mask mAP50-95", "Value": "N/A", "Conclusion": "STOP_TRANSFER_HARD_GATE"},
    ],
    "Experiment",
)

matrix = [
    ["Baseline", "Baseline", "PASS_WITH_NUMERICAL_WAIVER", "VAL mask mAP50-95 0.298981", "YES", "Required reference"],
    ["Low-confidence", "Historical", "POSITIVE", "91/173 false negatives recoverable", "NO_MODEL", "Diagnostic only"],
    ["One-class", "Historical", "NO_CLEAR_GAIN", "Crack AP50-95 0.055484", "NO", "No clear gain"],
    ["Hard Mining", "Historical", "POSITIVE_CANDIDATE", "VAL mask mAP50-95 0.311318", "YES", "Best completed segmentation candidate"],
    ["ROI CE", "Historical", "COMPLETE", "VAL classifier macro F1 0.67780", "NO_SEG_MODEL", "Different metric domain"],
    ["SupCon", "Historical", "SUPCON_POSITIVE", "VAL classifier macro F1 0.69370", "NO_SEG_MODEL", "Stage2 was negative"],
    ["Stage2", "Historical", "NEGATIVE", "Best mask F1 below YOLO at conf 0.25", "NO", "End-to-end degradation"],
    ["AUX CE", "Historical reconstruction", "SKIPPED_BY_ENGINEERING_GATE", "batch32 smoke over 70 seconds", "NO", "No comparable formal result"],
    ["KD", "Historical reconstruction", "SKIPPED_BY_ENGINEERING_GATE", "same online-teacher bottleneck", "NO", "No comparable formal result"],
    ["SimSiam", "New extension", "STOP_TRANSFER_HARD_GATE", "SSL PASS; reload exact 160/240", "NO", "Backbone transfer not proven"],
]
fields = ["Method", "Historical_or_extension", "Result", "Primary_evidence", "Keep_for_final_verify", "Reason"]
write_csv(FAST / "method_status_matrix.csv", [dict(zip(fields, row)) for row in matrix], fields)

summary_fields = ["Experiment", "Formal_training", "VAL", "Status", "Test_accessed", "Evidence"]
summary_rows = [
    dict(zip(summary_fields, ["Exp08 AUX_CE", "NO", "NO", "SKIPPED_BY_ENGINEERING_GATE", "false", "exp08_kd/summary.json"])),
    dict(zip(summary_fields, ["Exp08 KD", "NO", "NO", "SKIPPED_BY_ENGINEERING_GATE", "false", "exp08_kd/summary.json"])),
    dict(zip(summary_fields, ["Exp09 SSL", "YES (100 epochs)", "N/A", "PASS_NO_COLLAPSE", "false", "exp09_simsiam/ssl/summary.json"])),
    dict(zip(summary_fields, ["Exp09 downstream", "NO", "NO", "STOP_TRANSFER_HARD_GATE", "false", "exp09_simsiam/transfer_report.json"])),
]
write_csv(FAST / "fasttrack_c_summary.csv", summary_rows, summary_fields)

(ROOT / "docs/exp08_kd_fast_repro.md").write_text("""# Exp08 classifier-teacher KD fast reproduction

Status: **SKIPPED_BY_ENGINEERING_GATE**. `test_accessed=false`.

This is a historical-method reconstruction, not a restoration of original code. The teacher is the frozen Exp06 SupCon ResNet18 checkpoint with SHA256 `8e22f17c029eb0f3cb9416673a3503e0d37c7b98b91f126da2107e23fe58c32b`. The implementation puts the teacher in evaluation mode and disables gradients. The student starts from official `yolo11n-seg.pt`.

Graph inspection selected layer 4 (`C3k2`): stride 8, 128 channels, and `[B,128,80,80]` at image size 640. Ground-truth boxes map to P3 ROIAlign 7x7, pooling, and a seven-class auxiliary head. Teacher ROIs use 1.2x boxes resized to 224x224 with ImageNet normalization; KD uses only the renormalized seven defect logits.

Smoke testing found and fixed a finite-check implementation bug. Online teacher ROI extraction then exhausted memory at batch 4; chunking teacher crops by 16 made batch 32 forward/loss finite, but one training step exceeded 70 seconds. Completing both 100-epoch runs would require a teacher-target cache or data-loader redesign outside the Fast Repro boundary. Under task rule 5.3, AUX_CE and KD formal training, VAL comparison, lambda/T search, and checkpoint selection were not run. The test split was never accessed.
""", encoding="utf-8")

(ROOT / "docs/exp09_simsiam_fast_repro.md").write_text(f"""# Exp09 SimSiam fast reproduction

Status: **STOP / BACKBONE_TRANSFER_HARD_GATE**. `test_accessed=false`.

SSL used only the 668 frozen TRAIN images; VAL and TEST images seen were both zero. The encoder was official YOLO11n-seg layers 0-10 only, excluding neck and head ({ssl['encoder_parameter_count']} parameters, 240 state tensors). Frozen settings were batch 32, image size 512, 100 epochs, SGD momentum 0.9, weight decay 1e-4, learning rate 0.00625, and seed 42.

All 100 SSL epochs completed. Loss and checkpoint values were finite. Minimum feature std was {ssl['feature_std_min']:.6f}; minimum projector embedding std was {ssl['embedding_std_min']:.6f}; no collapse was detected. Adapted-backbone SHA256: `{ssl['adapted_backbone_sha256']}`.

Transfer key audit reported expected=240, missing=0, unexpected=0, and 120 tensors changed relative to COCO. After saving and reloading the downstream YOLO checkpoint, however, only 160/240 backbone tensors were byte-for-byte equal to the SimSiam export. Example tensor `0.bn.running_mean` had COCO/SimSiam/reloaded hashes `{transfer['coco_hash']}` / `{transfer['simsiam_hash']}` / `{transfer['downstream_hash']}`. Complete backbone migration therefore was not proven and task Hard Gate 10 fired.

Per the gate, downstream 100-epoch training and VAL evaluation were not run. No classification of SimSiam as positive/no-clear-gain/negative is valid. No repair search was performed, and TEST was never accessed.
""", encoding="utf-8")

(ROOT / "docs/handoffs/FASTTRACK_C_REVIEW.md").write_text("""# FastTrack-C review

FastTrack-C did not complete: Exp08 was skipped by its engineering gate, while Exp09 completed 100 SSL epochs without collapse but stopped at the backbone-transfer hard gate. Only 160/240 exported backbone tensors were byte-identical after downstream checkpoint reload, so downstream training and VAL were forbidden. `test_accessed=false` throughout.

The completed segmentation candidates remain Baseline and Exp05 Hard Mining. The old-method recommendation is Exp05 Hard Mining; no extension candidate qualifies. Do not enter Exp10 or Exp11 without user review and explicit authorization.

Evidence: `results/fast_repro/fasttrack_c_summary.csv`, `results/fast_repro/method_status_matrix.csv`, `results/fast_repro/fast_repro_master_summary.csv`, and `results/fast_repro/figures/exp09_simsiam/`.
""", encoding="utf-8")

index_path = ROOT / "docs/experiment_index.md"
index = index_path.read_text(encoding="utf-8")
for line in [
    "| Exp08 | Classifier Teacher to YOLO KD | SKIPPED_BY_ENGINEERING_GATE | `docs/exp08_kd_fast_repro.md` |",
    "| Exp09 | SimSiam YOLO backbone adaptation | STOP / TRANSFER_HARD_GATE | `docs/exp09_simsiam_fast_repro.md` |",
]:
    if line not in index:
        index += "\n" + line
index_path.write_text(index.rstrip() + "\n", encoding="utf-8")

state_path = ROOT / "docs/PROJECT_STATE.md"
state = state_path.read_text(encoding="utf-8")
state = state.replace("FastTrack-B complete; stopped for review", "FastTrack-C stopped at Exp09 backbone-transfer Hard Gate")
state = state.replace("Exp07 Stage2 (NEGATIVE); Exp06 SupCon positive as ROI classifier", "Exp09 SSL PASS/no collapse; downstream blocked by transfer verification")
state = state.replace("Exp02 Baseline Gate PASS_WITH_NUMERICAL_WAIVER; FastTrack-A complete", "Exp09 BACKBONE_TRANSFER_HARD_GATE; FastTrack-C stopped")
state = state.replace("None automatically; FastTrack-C requires explicit authorization", "None; review Exp09 transfer Hard Gate before any further experiment")
state = state.replace("Wait for user review of `docs/handoffs/FASTTRACK_B_REVIEW.md`. Do not start FastTrack-C or access test automatically.", "Wait for user review of `docs/handoffs/FASTTRACK_C_REVIEW.md`. Do not start Exp10/Exp11 or access test automatically.")
state = state.replace("2026-08-13T15:00:00+08:00", "2026-08-13T18:45:00+08:00")
state_path.write_text(state, encoding="utf-8")

roadmap_path = ROOT / "ROADMAP.md"
roadmap = roadmap_path.read_text(encoding="utf-8")
roadmap = roadmap.replace("- [ ] FastTrack-C KD / SimSiam", "- [x] Exp08 KD implementation/smoke: SKIPPED_BY_ENGINEERING_GATE\n- [x] Exp09 SimSiam SSL 100 epochs: PASS / no collapse\n- [ ] Exp09 downstream: STOP_BY_BACKBONE_TRANSFER_HARD_GATE\n- [ ] Exp10/Exp11: forbidden pending review\n- [ ] FastTrack-C KD / SimSiam")
roadmap_path.write_text(roadmap, encoding="utf-8")

registry_path = ROOT / "results/experiment_registry.csv"
registry = list(csv.DictReader(registry_path.open(encoding="utf-8-sig")))
registry_fields = list(registry[0])
registry = [row for row in registry if row["experiment_id"] not in {"Exp08", "Exp09"}]
registry.extend([
    dict(zip(registry_fields, ["Exp08", "Classifier Teacher to YOLO KD", "SKIPPED_BY_ENGINEERING_GATE", "2026-08-13T09:20:00Z", "2026-08-13T09:30:00Z", "720ed34", "/root/autodl-tmp/borescope-new-seg-data/v1", "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d", "bash scripts/exp08_kd_fast_repro.sh", "results/fast_repro/exp08_kd", "Online teacher ROI batch32 exceeded 70 seconds per batch; no formal training; test untouched"])),
    dict(zip(registry_fields, ["Exp09", "SimSiam YOLO backbone adaptation", "STOP_TRANSFER_HARD_GATE", "2026-08-13T09:31:00Z", "2026-08-13T10:35:00Z", "720ed34", "/root/autodl-tmp/borescope-new-seg-data/v1", "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d", "bash scripts/exp09_simsiam_fast_repro.sh", "results/fast_repro/exp09_simsiam", "SSL 100 epochs PASS/no collapse; reload exact 160/240; downstream not run; test untouched"])),
])
write_csv(registry_path, registry, registry_fields)

changelog_path = ROOT / "CHANGELOG.md"
changelog = changelog_path.read_text(encoding="utf-8")
entry = "- FastTrack-C stopped at the Exp09 backbone-transfer Hard Gate: Exp08 was skipped by its engineering gate; Exp09 SSL completed without collapse, but only 160/240 tensors were exact after downstream reload.\n- Exp09 downstream/VAL, Exp10, and Exp11 were not run; `test_accessed=false`.\n"
if entry not in changelog:
    changelog = changelog.replace("## Unreleased\n", "## Unreleased\n\n" + entry)
changelog_path.write_text(changelog, encoding="utf-8")

manifest_path = FAST / "artifact_manifest.csv"
manifest = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
manifest_fields = list(manifest[0])
paths = {row["file_path"] for row in manifest}
for png in sorted(FIG.glob("*.png")):
    image = cv2.imread(str(png))
    if image is None or image.size == 0:
        raise RuntimeError(f"OpenCV decode failed: {png}")
    rel = str(png.relative_to(ROOT))
    if rel not in paths:
        manifest.append({
            "experiment": "fasttrack_c",
            "artifact_type": "png",
            "file_path": rel,
            "file_size": str(png.stat().st_size),
            "decode_pass": "True",
        })
write_csv(manifest_path, manifest, manifest_fields)

report = {
    "status": "STOP_TRANSFER_HARD_GATE",
    "test_accessed": False,
    "artifact_count": len(manifest),
    "artifact_decode_pass": True,
    "ssl_summary_sha256": hashlib.sha256((EXP09 / "ssl/summary.json").read_bytes()).hexdigest(),
    "transfer_report_sha256": hashlib.sha256((EXP09 / "transfer_report.json").read_bytes()).hexdigest(),
}
(FAST / "fasttrack_c_gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
