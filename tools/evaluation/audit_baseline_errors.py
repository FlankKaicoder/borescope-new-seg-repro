#!/usr/bin/env python3
"""Exp02.2 train-derived size bins and fixed-point validation error audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


CONF = 0.25
NMS_IOU = 0.70
MATCH_IOU = 0.50
SIZE_NAMES = ("tiny", "small", "medium", "large")


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row["split"] for row in rows} - {"train", "val", "test"}:
        raise RuntimeError("Unexpected split value in manifest")
    return rows


def label_instances(path: Path, width: int, height: int) -> list[dict]:
    instances = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = raw.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            raise RuntimeError(f"Invalid segmentation row {path}:{line_no}")
        cls = int(values[0])
        coords = np.asarray([float(x) for x in values[1:]], dtype=np.float32).reshape(-1, 2)
        points = np.rint(coords * np.asarray([width - 1, height - 1])).astype(np.int32)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [points], 1)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            box = np.zeros(4, np.float32)
        else:
            box = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
        instances.append({"class_id": cls, "mask": mask.astype(bool), "box": box,
                          "relative_mask_area": float(mask.sum() / (width * height))})
    return instances


def image_and_label(root: Path, row: dict[str, str]) -> tuple[Path, Path]:
    split, stem, suffix = row["split"], row["stem"], row["image_suffix"]
    return root / "images" / split / f"{stem}{suffix}", root / "labels" / split / f"{stem}.txt"


def size_bin(area: float, q: np.ndarray) -> str:
    if area <= q[0]: return SIZE_NAMES[0]
    if area <= q[1]: return SIZE_NAMES[1]
    if area <= q[2]: return SIZE_NAMES[2]
    return SIZE_NAMES[3]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return float(inter / union) if union else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def prediction_instances(result, height: int, width: int) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    if result.masks is None:
        masks = [np.zeros((height, width), dtype=bool) for _ in classes]
    else:
        masks = []
        for raw in result.masks.data.cpu().numpy():
            if raw.shape != (height, width):
                raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
            masks.append(raw >= 0.5)
    return [{"class_id": int(c), "confidence": float(score), "box": boxes[i].astype(np.float32), "mask": masks[i]}
            for i, (c, score) in enumerate(zip(classes, confs))]


def draw_visual(path: Path, image: np.ndarray, stem: str, categories: set[str], gts: list[dict], preds: list[dict], names: dict[int, str]) -> None:
    canvas = image.copy()
    for gt in gts:
        contours, _ = cv2.findContours(gt["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 210, 0), 2)
        x, y = int(gt["box"][0]), int(gt["box"][1])
        cv2.putText(canvas, f"GT {names[gt['class_id']]}", (x, max(18, y)), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 210, 0), 1, cv2.LINE_AA)
    for pred in preds:
        contours, _ = cv2.findContours(pred["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 230), 2)
        x, y = int(pred["box"][0]), int(pred["box"][1])
        cv2.putText(canvas, f"P {names[pred['class_id']]} {pred['confidence']:.2f}", (x, min(canvas.shape[0] - 5, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 230), 1, cv2.LINE_AA)
    title = f"{stem} | " + ",".join(sorted(categories)) + " | GT=green PRED=red"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def summarize_group(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in (*train_rows, *val_rows):
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        gt = len(items); tp = sum(x["error_type"] == "TP" for x in items)
        entry = dict(zip(keys, key))
        matched_ious = [x["best_mask_iou"] for x in items if x["error_type"] == "TP"]
        entry.update(gt_instances=gt, true_positives=tp, false_negatives=gt - tp,
                     recall_at_fixed_point=tp / gt if gt else 0.0,
                     mean_matched_mask_iou=float(np.mean(matched_ious)) if tp else "",
                     median_matched_mask_iou=float(np.median(matched_ious)) if tp else "")
        out.append(entry)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-visuals-per-type", type=int, default=8)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    visuals = args.output / "qualitative_errors"; visuals.mkdir()

    config = yaml.safe_load((args.data_root / "data.yaml").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in config["names"].items()}
    rows = load_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    group_label_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        near_id = row.get("near_duplicate_group_id", "").strip()
        if near_id:
            group_label_sets[near_id].add(row.get("labels_present", "").strip())
    # The test rows are deliberately neither resolved to paths nor passed to the model.

    train_areas = []
    for row in train_rows:
        _, label = image_and_label(args.data_root, row)
        for gt in label_instances(label, int(row["image_width"]), int(row["image_height"])):
            train_areas.append(gt["relative_mask_area"])
    quartiles = np.quantile(np.asarray(train_areas), [.25, .50, .75])
    write_json(args.output / "train_area_thresholds.json", {
        "derivation_split": "train", "train_instances": len(train_areas),
        "relative_mask_area_definition": "native-resolution filled polygon pixels / image pixels",
        "q25": quartiles[0], "q50": quartiles[1], "q75": quartiles[2], "bin_policy": "right-closed quartiles",
    })

    val_paths = [image_and_label(args.data_root, row)[0] for row in val_rows]
    model = YOLO(str(args.weights))
    predictions = model.predict(source=[str(x) for x in val_paths], imgsz=640, conf=CONF, iou=NMS_IOU,
                                batch=args.batch, device=0, retina_masks=True, verbose=False, stream=True)
    gt_audit, pred_audit = [], []
    visual_count = Counter()
    for row, image_path, result in zip(val_rows, val_paths, predictions, strict=True):
        image = cv2.imread(str(image_path))
        if image is None: raise RuntimeError(f"Unreadable validation image {image_path}")
        h, w = image.shape[:2]
        _, label_path = image_and_label(args.data_root, row)
        gts = label_instances(label_path, w, h)
        preds = prediction_instances(result, h, w)
        miou = np.zeros((len(gts), len(preds)), np.float32)
        biou = np.zeros_like(miou)
        for gi, gt in enumerate(gts):
            for pi, pred in enumerate(preds):
                miou[gi, pi] = mask_iou(gt["mask"], pred["mask"])
                biou[gi, pi] = box_iou(gt["box"], pred["box"])
        candidates = sorted(((float(miou[g, p]), g, p) for g in range(len(gts)) for p in range(len(preds)) if miou[g, p] >= MATCH_IOU), reverse=True)
        used_g, used_p, spatial_matches = set(), set(), {}
        for score, gi, pi in candidates:
            if gi not in used_g and pi not in used_p:
                used_g.add(gi); used_p.add(pi); spatial_matches[gi] = pi

        image_categories = set()
        near_id = row.get("near_duplicate_group_id", "").strip()
        duplicate_context = ("singleton" if not near_id else
                             "near_cross_label" if len(group_label_sets[near_id]) > 1 else "near_same_label")
        for gi, gt in enumerate(gts):
            pi = spatial_matches.get(gi)
            best_m = float(miou[gi].max()) if len(preds) else 0.0
            best_b = float(biou[gi].max()) if len(preds) else 0.0
            matched_conf, pred_class = "", ""
            if pi is not None:
                pred = preds[pi]; matched_conf = pred["confidence"]; pred_class = pred["class_id"]
                error = "TP" if pred["class_id"] == gt["class_id"] else "wrong_class"
                best_m, best_b = float(miou[gi, pi]), float(biou[gi, pi])
            else:
                same = [p for p, pred in enumerate(preds) if pred["class_id"] == gt["class_id"]]
                if same:
                    best_same = max(same, key=lambda p: float(miou[gi, p]))
                    best_m, best_b = float(miou[gi, best_same]), float(biou[gi, best_same])
                    matched_conf, pred_class = preds[best_same]["confidence"], preds[best_same]["class_id"]
                    error = "low_mask_iou" if best_b >= MATCH_IOU else "low_box_iou"
                else:
                    error = "FN_no_same_class_candidate"
            if error != "TP": image_categories.add(error)
            gt_audit.append({"stem": row["stem"], "near_duplicate_group_id": near_id,
                             "is_near_duplicate_member": bool(near_id), "gt_index": gi,
                             "duplicate_context": duplicate_context,
                             "class_id": gt["class_id"], "class_name": names[gt["class_id"]],
                             "relative_mask_area": gt["relative_mask_area"], "size_bin": size_bin(gt["relative_mask_area"], quartiles),
                             "error_type": error, "pred_class_id": pred_class, "matched_confidence": matched_conf,
                             "best_mask_iou": best_m, "best_box_iou": best_b})
        matched_pred = set(spatial_matches.values())
        for pi, pred in enumerate(preds):
            matched_gt = next((g for g, pidx in spatial_matches.items() if pidx == pi), None)
            if matched_gt is not None and pred["class_id"] == gts[matched_gt]["class_id"]:
                error = "TP"
            elif matched_gt is not None:
                error = "wrong_class_FP"
            else:
                error = "FP_unmatched"
            if error != "TP": image_categories.add(error)
            pred_audit.append({"stem": row["stem"], "near_duplicate_group_id": near_id,
                               "duplicate_context": duplicate_context,
                               "pred_index": pi, "class_id": pred["class_id"], "class_name": names[pred["class_id"]],
                               "confidence": pred["confidence"], "error_type": error,
                               "matched_gt_index": "" if matched_gt is None else matched_gt})
        for category in sorted(image_categories):
            if visual_count[category] < args.max_visuals_per_type:
                safe = category.replace("/", "_")
                draw_visual(visuals / f"{safe}__{row['stem']}.jpg", image, row["stem"], image_categories, gts, preds, names)
                visual_count[category] += 1

    write_csv(args.output / "gt_instance_audit.csv", gt_audit)
    write_csv(args.output / "prediction_audit.csv", pred_audit)
    write_csv(args.output / "metrics_by_size.csv", summarize_group(gt_audit, ("size_bin",)))
    write_csv(args.output / "metrics_by_class_and_size.csv", summarize_group(gt_audit, ("class_id", "class_name", "size_bin")))

    gt_errors = Counter(row["error_type"] for row in gt_audit)
    pred_errors = Counter(row["error_type"] for row in pred_audit)
    tp = gt_errors["TP"]
    wrong = gt_errors["wrong_class"]
    summary = {
        "status": "PASS", "weights": str(args.weights), "weights_sha256": sha256(args.weights),
        "evaluation_split": "val", "test_accessed": False,
        "diagnostic_operating_point": {"confidence": CONF, "nms_iou": NMS_IOU, "mask_match_iou": MATCH_IOU},
        "val_images": len(val_rows), "gt_instances": len(gt_audit), "pred_instances": len(pred_audit),
        "TP": tp, "FN": len(gt_audit) - tp, "FP": len(pred_audit) - tp,
        "wrong_class": wrong, "low_mask_iou": gt_errors["low_mask_iou"], "low_box_iou": gt_errors["low_box_iou"],
        "fn_no_same_class_candidate": gt_errors["FN_no_same_class_candidate"],
        "gt_error_type_counts": gt_errors, "prediction_error_type_counts": pred_errors,
        "visual_counts_by_error_type": visual_count,
        "interpretation_limit": "Fixed-point diagnostic only; no low-confidence recovery claim and no threshold tuning.",
    }
    write_json(args.output / "error_summary.json", summary)

    near_rows = []
    for context in ("singleton", "near_same_label", "near_cross_label"):
        subset = [x for x in gt_audit if x["duplicate_context"] == context]
        errors = sum(x["error_type"] != "TP" for x in subset)
        wrong = sum(x["error_type"] == "wrong_class" for x in subset)
        near_rows.append({"duplicate_context": context, "gt_instances": len(subset), "errors": errors,
                          "wrong_class": wrong,
                          "error_rate": errors / len(subset) if subset else 0.0})
    write_csv(args.output / "near_duplicate_error_concentration.csv", near_rows)
    write_json(args.output / "run_summary.json", summary | {"train_area_quartiles": quartiles.tolist()})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
