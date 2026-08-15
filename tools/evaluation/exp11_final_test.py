#!/usr/bin/env python3
"""One-time frozen Exp11 TEST evaluation; no selection, sweep, or training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from exp10_unified_val import metric_record
from fasttrack_common import NAMES, biou, class_matches, image_label, label_instances, miou, predictions, rows

EXPECTED_SHA = "2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9"
SIZE_Q = (0.0020196759259259256, 0.007364969135802469, 0.029209280303030303)
SIZE_NAMES = ("tiny", "small", "medium", "large")
CONF = 0.25
NMS_IOU = 0.70
MASK_IOU = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not records and fields is None:
        raise ValueError(f"No records for {path}")
    names = fields or list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def size_bin(area: float) -> str:
    if area <= SIZE_Q[0]:
        return "tiny"
    if area <= SIZE_Q[1]:
        return "small"
    if area <= SIZE_Q[2]:
        return "medium"
    return "large"


def draw_overlay(image: np.ndarray, gts: list[dict], preds: list[dict], title: str) -> np.ndarray:
    canvas = image.copy()
    for gt in gts:
        contours, _ = cv2.findContours(gt["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 220, 0), 2)
    for pred in preds:
        contours, _ = cv2.findContours(pred["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 230), 2)
        x, y = map(int, pred["box"][:2])
        cv2.putText(canvas, f"{NAMES[pred['class_id']]} {pred['confidence']:.2f}", (x, max(18, y)), 0, 0.45, (0, 0, 230), 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(canvas, title[:120], (5, 22), 0, 0.48, (255, 255, 255), 1)
    return canvas


def make_grid(paths: list[Path], output: Path) -> None:
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot decode qualitative image: {path}")
        thumbs.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
    cols = 8
    rows_n = (len(thumbs) + cols - 1) // cols
    blank = np.zeros((240, 320, 3), np.uint8)
    while len(thumbs) < rows_n * cols:
        thumbs.append(blank.copy())
    grid = np.vstack([np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)])
    cv2.imwrite(str(output), grid)


def verify_artifacts(output: Path) -> dict[str, Any]:
    image_paths = sorted([*output.rglob("*.png"), *output.rglob("*.jpg"), *output.rglob("*.jpeg")])
    decode_failures = []
    for path in image_paths:
        if path.stat().st_size <= 0 or cv2.imread(str(path)) is None:
            decode_failures.append(str(path))
    csv_paths = sorted(output.rglob("*.csv"))
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            list(csv.reader(handle))
    json_paths = sorted(output.rglob("*.json"))
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "PASS" if not decode_failures else "FAIL",
        "checked_images": len(image_paths),
        "all_images_exist_nonempty_decode": not decode_failures,
        "decode_failures": decode_failures,
        "parsed_csv": len(csv_paths),
        "parsed_json": len(json_paths),
        "test_accessed": True,
        "model_selection_closed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "PASS" or freeze.get("test_accessed_before_freeze") is not False:
        raise RuntimeError("EXP11_EVALUATION_INVALIDATING_GATE: invalid Candidate Freeze")
    weights = args.weights.resolve()
    if not weights.is_file() or weights.stat().st_size <= 0 or sha256(weights) != EXPECTED_SHA:
        raise RuntimeError("CHECKPOINT_FREEZE_HARD_GATE")
    if str(weights) != freeze["checkpoint_absolute_path"] or freeze["checkpoint_sha256"] != EXPECTED_SHA:
        raise RuntimeError("CHECKPOINT_FREEZE_HARD_GATE: freeze/checkpoint mismatch")

    model = YOLO(str(weights))
    nonfinite = [name for name, tensor in model.model.state_dict().items() if torch.is_tensor(tensor) and not torch.isfinite(tensor).all()]
    if nonfinite:
        raise RuntimeError(f"CHECKPOINT_FREEZE_HARD_GATE nonfinite={nonfinite[:10]}")

    output.mkdir(parents=True)
    artifacts = output / "artifacts"
    qualitative = output / "qualitative"
    qualitative.mkdir()
    shutil.copy2(args.freeze, output / "candidate_freeze.json")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "checkpoint_sha256.txt").write_text(f"{EXPECTED_SHA}  {weights}\n", encoding="utf-8")
    dump_json(output / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "test_accessed": True,
    })

    # First and only formal standard TEST metrics for the frozen candidate.
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=640,
        batch=32,
        device=0,
        plots=True,
        project=str(artifacts / "ultralytics"),
        name="final_seed44_baseline",
        exist_ok=False,
        verbose=True,
    )
    box = metric_record(metrics.box)
    mask = metric_record(metrics.seg)

    manifest_rows = [row for row in rows(args.manifest.resolve()) if row["split"] == "test"]
    if len(manifest_rows) != 147 or sum(int(row["instance_count"]) for row in manifest_rows) != 285:
        raise RuntimeError("EXP11_EVALUATION_INVALIDATING_GATE: frozen TEST manifest mismatch")
    test_paths = [image_label(args.data_root.resolve(), row)[0] for row in manifest_rows]
    fixed_stream = model.predict(
        source=[str(path) for path in test_paths], imgsz=640, conf=CONF, iou=NMS_IOU,
        batch=32, device=0, retina_masks=True, verbose=False, stream=True,
    )

    total = Counter()
    per_class = defaultdict(Counter)
    size_total = Counter()
    size_hit = Counter()
    error_records: list[dict[str, Any]] = []
    image_cases: list[dict[str, Any]] = []

    for row, result in zip(manifest_rows, fixed_stream, strict=True):
        image_path, label_path = image_label(args.data_root.resolve(), row)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"EXP11_EVALUATION_INVALIDATING_GATE: unreadable image {image_path}")
        height, width = image.shape[:2]
        gts = label_instances(label_path, width, height)
        preds = predictions(result, height, width)
        matches = class_matches(gts, preds, MASK_IOU)
        matched_pred = set(matches.values())
        total.update(GT=len(gts), TP=len(matches), PRED=len(preds))
        categories: set[str] = set()
        if len(gts) > 1:
            categories.add("MULTI_DEFECT_SCENE")

        for gt_index, gt in enumerate(gts):
            cls = gt["class_id"]
            area = float(gt["mask"].sum() / (height * width))
            bin_name = size_bin(area)
            size_total[bin_name] += 1
            per_class[cls]["GT"] += 1
            base = {
                "stem": row["stem"], "image_path": str(image_path), "gt_index": gt_index,
                "gt_class_id": cls, "gt_class_name": NAMES[cls], "size_bin": bin_name,
                "multi_defect_scene": len(gts) > 1,
            }
            if gt_index in matches:
                pred_index = matches[gt_index]
                pred = preds[pred_index]
                score = miou(gt["mask"], pred["mask"])
                size_hit[bin_name] += 1
                per_class[cls]["TP"] += 1
                category = "POOR_MASK_BOUNDARY" if score < 0.65 else "TP"
                categories.update({"TP", category})
                error_records.append({**base, "pred_index": pred_index, "pred_class_id": pred["class_id"],
                    "pred_class_name": NAMES[pred["class_id"]], "confidence": pred["confidence"],
                    "mask_iou": score, "box_iou": biou(gt["box"], pred["box"]), "category": category,
                    "notes": "matched at frozen conf=0.25"})
                continue

            per_class[cls]["FN"] += 1
            same = [(idx, miou(gt["mask"], pred["mask"]), biou(gt["box"], pred["box"]))
                    for idx, pred in enumerate(preds) if pred["class_id"] == cls]
            wrong = [(idx, miou(gt["mask"], pred["mask"]), biou(gt["box"], pred["box"]))
                     for idx, pred in enumerate(preds) if pred["class_id"] != cls]
            spatial_wrong = [item for item in wrong if item[1] >= MASK_IOU or item[2] >= 0.50]
            if spatial_wrong:
                selected = max(spatial_wrong, key=lambda item: max(item[1], item[2]))
                category = "WRONG_CLASS"
            elif same:
                selected = max(same, key=lambda item: item[1])
                category = "LOCALIZATION_FAILURE"
            else:
                selected = None
                category = "NO_RESPONSE"
            categories.add(category)
            if bin_name in ("tiny", "small"):
                categories.add("TINY_SMALL_MISS")
            pred = None if selected is None else preds[selected[0]]
            error_records.append({**base, "pred_index": "" if selected is None else selected[0],
                "pred_class_id": "" if pred is None else pred["class_id"],
                "pred_class_name": "" if pred is None else NAMES[pred["class_id"]],
                "confidence": "" if pred is None else pred["confidence"],
                "mask_iou": "" if selected is None else selected[1], "box_iou": "" if selected is None else selected[2],
                "category": category, "notes": "LOW_CONF status NOT_AVAILABLE; no TEST threshold sweep performed"})

        for pred_index, pred in enumerate(preds):
            cls = pred["class_id"]
            per_class[cls]["PRED"] += 1
            if pred_index in matched_pred:
                continue
            per_class[cls]["FP"] += 1
            categories.add("FP")
            error_records.append({
                "stem": row["stem"], "image_path": str(image_path), "gt_index": "",
                "gt_class_id": "", "gt_class_name": "", "size_bin": "N/A",
                "multi_defect_scene": len(gts) > 1, "pred_index": pred_index,
                "pred_class_id": cls, "pred_class_name": NAMES[cls], "confidence": pred["confidence"],
                "mask_iou": "", "box_iou": "", "category": "FP", "notes": "unmatched prediction at frozen conf=0.25",
            })
        image_cases.append({"row": row, "image": image, "gts": gts, "preds": preds, "categories": categories})

    total["FP"] = total["PRED"] - total["TP"]
    total["FN"] = total["GT"] - total["TP"]
    fixed_p = total["TP"] / total["PRED"] if total["PRED"] else 0.0
    fixed_r = total["TP"] / total["GT"] if total["GT"] else 0.0
    fixed_f1 = 2 * fixed_p * fixed_r / (fixed_p + fixed_r) if fixed_p + fixed_r else 0.0

    overall_rows = [
        {"domain": "Box", **box},
        {"domain": "Mask", **mask},
    ]
    write_csv(output / "overall_metrics.csv", overall_rows)

    per_class_rows = []
    names = metrics.names
    for class_id in range(7):
        per_class_rows.append({
            "class_id": class_id, "class_name": names[class_id], "GT": per_class[class_id]["GT"],
            "Box_P": metric_record(metrics.box, class_id)["P"], "Box_R": metric_record(metrics.box, class_id)["R"],
            "Box_AP50": metric_record(metrics.box, class_id)["mAP50"], "Box_AP50_95": metric_record(metrics.box, class_id)["mAP50_95"],
            "Mask_P": metric_record(metrics.seg, class_id)["P"], "Mask_R": metric_record(metrics.seg, class_id)["R"],
            "Mask_AP50": metric_record(metrics.seg, class_id)["mAP50"], "Mask_AP50_95": metric_record(metrics.seg, class_id)["mAP50_95"],
        })
    write_csv(output / "per_class_metrics.csv", per_class_rows)

    fixed_rows = [{"scope": "OVERALL", "class_id": "ALL", "class_name": "ALL", "GT": total["GT"],
                   "TP": total["TP"], "FP": total["FP"], "FN": total["FN"],
                   "Precision": fixed_p, "Recall": fixed_r, "F1": fixed_f1}]
    for class_id in range(7):
        counts = per_class[class_id]
        precision = counts["TP"] / counts["PRED"] if counts["PRED"] else 0.0
        recall = counts["TP"] / counts["GT"] if counts["GT"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fixed_rows.append({"scope": "CLASS", "class_id": class_id, "class_name": NAMES[class_id], "GT": counts["GT"],
                           "TP": counts["TP"], "FP": counts["FP"], "FN": counts["FN"],
                           "Precision": precision, "Recall": recall, "F1": f1})
    write_csv(output / "fixed_threshold_metrics.csv", fixed_rows)

    size_rows = [{"size_bin": name, "GT": size_total[name], "TP": size_hit[name],
                  "FN": size_total[name] - size_hit[name],
                  "Recall": size_hit[name] / size_total[name] if size_total[name] else 0.0}
                 for name in SIZE_NAMES]
    write_csv(output / "size_metrics.csv", size_rows)
    write_csv(output / "error_instances.csv", error_records)

    # Deterministic representative selection without changing evaluator settings.
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    order = ("WRONG_CLASS", "LOCALIZATION_FAILURE", "NO_RESPONSE", "POOR_MASK_BOUNDARY",
             "TINY_SMALL_MISS", "MULTI_DEFECT_SCENE", "FP", "TP")
    for category in order:
        for case in image_cases:
            stem = case["row"]["stem"]
            if stem not in seen and category in case["categories"]:
                selected.append(case); seen.add(stem)
                if sum(category in item["categories"] for item in selected) >= 8:
                    break
    for case in image_cases:
        if len(selected) >= 64:
            break
        stem = case["row"]["stem"]
        if stem not in seen:
            selected.append(case); seen.add(stem)
    selected = selected[:64]
    qualitative_paths = []
    for index, case in enumerate(selected, 1):
        category_text = "+".join(sorted(case["categories"])) or "NO_EVENT"
        path = qualitative / f"{index:03d}_{case['row']['stem']}.jpg"
        overlay = draw_overlay(case["image"], case["gts"], case["preds"], f"{case['row']['stem']} {category_text}")
        cv2.imwrite(str(path), overlay)
        qualitative_paths.append(path)
    make_grid(qualitative_paths, artifacts / "baseline_test_qualitative_grid.jpg")

    hardest_ap = sorted(per_class_rows, key=lambda row: row["Mask_AP50_95"])[:3]
    hardest_recall = sorted(per_class_rows, key=lambda row: row["Mask_R"])[:3]
    taxonomy_counts = Counter(row["category"] for row in error_records)
    summary = {
        "status": "PASS", "phase": "EXP11_ONE_FINAL_FROZEN_EVALUATION",
        "test_accessed": True, "model_selection_closed": True, "post_test_training": False,
        "post_test_threshold_model_seed_selection": False, "checkpoint": str(weights),
        "checkpoint_sha256": EXPECTED_SHA, "seed": 44, "test_images": 147, "test_instances": 285,
        "Box": box, "Mask": mask, "fixed_threshold": fixed_rows[0], "size_metrics": size_rows,
        "selected_val_mask_map50_95": freeze["selected_val_mask_map50_95"],
        "test_minus_val_mask_map50_95": mask["mAP50_95"] - freeze["selected_val_mask_map50_95"],
        "hardest_by_mask_ap50_95": hardest_ap, "hardest_by_mask_recall": hardest_recall,
        "error_taxonomy_counts": dict(taxonomy_counts), "low_conf_test_taxonomy": "NOT_AVAILABLE_NO_THRESHOLD_SWEEP",
        "qualitative_image_count": len(qualitative_paths),
        "evaluation_semantics": "Exp10 unified Ultralytics semantics; split=test; imgsz=640; fixed conf=0.25 NMS IoU=0.70 mask match IoU=0.50",
    }
    dump_json(output / "summary.json", summary)
    verification = verify_artifacts(output)
    dump_json(output / "artifact_verification.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError("EXP11_EVALUATION_INVALIDATING_GATE: artifact verification failed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
