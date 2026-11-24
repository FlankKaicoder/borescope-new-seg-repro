"""Postprocess already computed TRAIN predictions, without model execution."""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils import LOGGER, nms, ops
from ultralytics.utils.metrics import box_iou, mask_iou

from tools.exp14.a11_full_eval import native_match_pairs


class NmsTimeoutError(RuntimeError):
    """A native timeout cannot be accepted as a complete diagnostic result."""


@contextmanager
def capture_nms_timeout():
    messages = []

    class Handler(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "nms time limit" in message.lower() and "exceeded" in message.lower():
                messages.append(message)

    handler = Handler(level=logging.WARNING)
    LOGGER.addHandler(handler)
    try:
        yield
        if messages:
            raise NmsTimeoutError("; ".join(messages))
    finally:
        LOGGER.removeHandler(handler)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap_instances(overlap, count):
    """Preserve the existing overlap-ID and target-row correspondence."""
    if overlap.ndim != 2 or not bool(torch.isfinite(overlap).all()):
        raise ValueError("Expected a finite two-dimensional overlap mask")
    if not bool((overlap == overlap.round()).all() and (overlap >= 0).all() and (overlap <= count).all()):
        raise ValueError("Overlap mask IDs are outside the image's target-row domain")
    indexes = torch.arange(1, count + 1, device=overlap.device).view(-1, 1, 1)
    return (overlap.unsqueeze(0) == indexes).float()


def coordinate_counts(boxes, height, width):
    if not len(boxes):
        return {"nonfinite": 0, "nonpositive_extent": 0, "outside_image_canvas": 0}
    valid = torch.isfinite(boxes).all(1)
    extent = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    outside = (boxes[:, 0] < 0) | (boxes[:, 1] < 0) | (boxes[:, 2] > width) | (boxes[:, 3] > height)
    return {"nonfinite": int((~valid).sum()), "nonpositive_extent": int(extent.sum()),
            "outside_image_canvas": int(outside.sum())}


def counts(gt, detections, box_pairs, mask_pairs):
    box_tp, mask_tp = len(box_pairs), len(mask_pairs)
    return {"gt": int(gt), "detections": int(detections),
            "box_tp": box_tp, "box_fp": detections - box_tp, "box_fn": gt - box_tp,
            "mask_tp": mask_tp, "mask_fp": detections - mask_tp, "mask_fn": gt - mask_tp,
            "box_precision": box_tp / detections if detections else None,
            "box_recall": box_tp / gt if gt else None,
            "mask_precision": mask_tp / detections if detections else None,
            "mask_recall": mask_tp / gt if gt else None}


def validate_inputs(decoded, proto, data):
    image = data["img"]
    if not isinstance(decoded, torch.Tensor) or not isinstance(proto, torch.Tensor):
        raise ValueError("Expected decoded/prototype tensors from the completed forward")
    if image.ndim != 4 or tuple(image.shape[1:]) != (3, 640, 640) or image.dtype != torch.float32:
        raise ValueError("Expected the frozen normalized TRAIN N3x640x640 float32 batch")
    if not bool(torch.isfinite(image).all() and (image >= 0).all() and (image <= 1).all()):
        raise ValueError("TRAIN image tensor is outside its normalized finite domain")
    batch = len(image)
    if proto.ndim != 4 or tuple(proto.shape[:1]) != (batch,) or tuple(proto.shape[-2:]) != (160, 160):
        raise ValueError("Prototype shape differs from the frozen mask resolution")
    if decoded.ndim != 3 or decoded.shape[0] != batch or decoded.shape[1] != 4 + 7 + proto.shape[1]:
        raise ValueError("Decoded BCN shape/class/mask-coefficient domain differs")
    if decoded.shape[-1] == 6:
        raise ValueError("Six-candidate shape would trigger the native end-to-end shortcut")
    if not bool(torch.isfinite(decoded).all() and torch.isfinite(proto).all()):
        raise ValueError("Nonfinite decoded predictions or prototype values")
    if decoded.dtype != torch.float32 or proto.dtype != torch.float32:
        raise ValueError("TRAIN postprocessing requires the frozen FP32 inference outputs")
    if decoded.device != proto.device:
        raise ValueError("Decoded predictions and prototypes use different devices")
    cls, boxes, batch_idx = data["cls"].reshape(-1), data["bboxes"], data["batch_idx"].reshape(-1)
    masks = data["masks"]
    if tuple(boxes.shape) != (len(cls), 4) or len(batch_idx) != len(cls) or tuple(masks.shape) != (batch, 160, 160):
        raise ValueError("GT class/box/image/overlap domains differ")
    if not bool(torch.isfinite(cls).all() and (cls == cls.round()).all() and (cls >= 0).all() and (cls < 7).all()):
        raise ValueError("GT classes differ from the frozen seven-class mapping")
    if not bool(torch.isfinite(batch_idx).all() and (batch_idx == batch_idx.round()).all()
                and (batch_idx >= 0).all() and (batch_idx < batch).all()):
        raise ValueError("Invalid GT image indexes")
    if not bool(torch.isfinite(boxes).all() and (boxes >= 0).all() and (boxes <= 1).all() and (boxes[:, 2:] > 0).all()):
        raise ValueError("Invalid normalized xywh GT boxes")


@torch.no_grad()
def postprocess(decoded, proto, data, output_dir, label):
    """Return fixed-conf TRAIN Box/Mask counts from existing raw predictions.

    Neither inference nor AP evaluation is performed. Native NMS is run on one
    cloned image prediction at a time, with its default timer and all scientific
    thresholds intact. A reported timer failure raises before committing NPZ.
    """
    validate_inputs(decoded, proto, data)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
        raise ValueError("Label must be a simple private artifact stem")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cls = data["cls"].reshape(-1).detach().to(decoded.device)
    normalized_boxes = data["bboxes"].detach().to(decoded.device)
    batch_idx = data["batch_idx"].reshape(-1).detach().to(decoded.device)
    overlap = data["masks"].detach().to(decoded.device)
    height, width = data["img"].shape[-2:]
    report = {}
    for confidence_key, confidence, suffix in ((".001", .001, "conf0001"), (".25", .25, "conf025")):
        per_image, arrays = [], {}
        per_class = [{"class_id": c, "gt": 0, "detections": 0, "box_tp": 0, "mask_tp": 0} for c in range(7)]
        gt_geometry_sum, pred_geometry_sum = 0., 0.
        for index in range(len(decoded)):
            # Pinned NMS modifies xywh to xyxy in-place on its prediction view.
            prediction = decoded[index:index + 1].detach().clone()
            with capture_nms_timeout():
                selected = nms.non_max_suppression(prediction, conf_thres=confidence, iou_thres=.7,
                    nc=7, multi_label=True, max_det=300, end2end=False)
            if len(selected) != 1:
                raise ValueError("Imagewise NMS returned an unexpected image count")
            pred = selected[0]
            invalid = coordinate_counts(pred[:, :4], height, width)
            if invalid["nonfinite"] or not bool(torch.isfinite(pred).all()):
                raise ValueError(f"Image{index} NMS produced nonfinite values: {invalid}")
            positions = batch_idx == index
            gt_cls = cls[positions]
            gt_boxes = ops.xywh2xyxy(normalized_boxes[positions]) * torch.tensor(
                [width, height, width, height], dtype=torch.float32, device=decoded.device)
            gt_masks = overlap_instances(overlap[index], len(gt_cls))
            predicted_masks = ops.process_mask(proto[index].detach(), pred[:, 6:], pred[:, :4],
                                               shape=(height, width), upsample=False)
            if tuple(predicted_masks.shape) != (len(pred), 160, 160):
                raise ValueError("Native predicted-mask shape changed")
            box_overlap = box_iou(gt_boxes, pred[:, :4])
            mask_overlap = mask_iou(gt_masks.flatten(1), predicted_masks.flatten(1).float())
            pred_classes, true_classes = pred[:, 5].cpu().numpy(), gt_cls.cpu().numpy()
            box_pairs = native_match_pairs(pred_classes, true_classes, box_overlap.cpu().numpy())
            mask_pairs = native_match_pairs(pred_classes, true_classes, mask_overlap.cpu().numpy())
            row = {"image_index": index, **counts(len(gt_cls), len(pred), box_pairs, mask_pairs),
                   "invalid_coordinates": invalid}
            if len(gt_cls) and len(pred):
                best_gt, best_pred = box_overlap.max(1).values, box_overlap.max(0).values
                gt_geometry_sum += float(best_gt.sum())
                pred_geometry_sum += float(best_pred.sum())
                row["gt_mean_best_box_iou_ignoring_class"] = float(best_gt.mean())
                row["prediction_mean_best_box_iou_ignoring_class"] = float(best_pred.mean())
            else:
                row["gt_mean_best_box_iou_ignoring_class"] = 0. if len(gt_cls) else None
                row["prediction_mean_best_box_iou_ignoring_class"] = 0. if len(pred) else None
            per_image.append(row)
            key = f"image_{index:03d}"
            arrays[key + "_predictions"] = pred.detach().cpu().numpy()
            arrays[key + "_box_matches"] = box_pairs
            arrays[key + "_mask_matches"] = mask_pairs
            for c in range(7):
                per_class[c]["gt"] += int((gt_cls == c).sum())
                per_class[c]["detections"] += int((pred[:, 5] == c).sum())
                per_class[c]["box_tp"] += sum(true_classes[p[0]] == c for p in box_pairs)
                per_class[c]["mask_tp"] += sum(true_classes[p[0]] == c for p in mask_pairs)
        total_gt = sum(row["gt"] for row in per_image)
        detections = sum(row["detections"] for row in per_image)
        total = {key: sum(row[key] for row in per_image) for key in ("gt", "detections", "box_tp", "box_fp", "box_fn", "mask_tp", "mask_fp", "mask_fn")}
        for kind in ("box", "mask"):
            total[kind + "_precision"] = total[kind + "_tp"] / detections if detections else None
            total[kind + "_recall"] = total[kind + "_tp"] / total_gt if total_gt else None
        for row in per_class:
            for kind in ("box", "mask"):
                row[kind + "_tp"] = int(row[kind + "_tp"])
                row[kind + "_fp"] = row["detections"] - row[kind + "_tp"]
                row[kind + "_fn"] = row["gt"] - row[kind + "_tp"]
        path = output_dir / f"{label}_{suffix}_nms.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report[confidence_key] = {"confidence": confidence, "images": len(decoded), **total,
            "per_image": per_image, "per_class": per_class,
            "gt_mean_best_box_iou_ignoring_class": gt_geometry_sum / total_gt if total_gt else None,
            "prediction_mean_best_box_iou_ignoring_class": pred_geometry_sum / detections if detections else None,
            "invalid_coordinates": {key: sum(row["invalid_coordinates"][key] for row in per_image)
                                    for key in ("nonfinite", "nonpositive_extent", "outside_image_canvas")},
            "nms_predictions": {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "array_keys": list(arrays), "columns": "xyxy,confidence,class,mask_coefficients"},
            "definitions": {"split": "FROZEN_AUGMENTED_TRAIN_ONLY", "new_mask_extension": True,
                "NMS_iou": .7, "max_det": 300, "multi_label": True, "end2end": False,
                "matching_iou": .5, "matching": "NATIVE_CLASS_AWARE_ONE_TO_ONE",
                "mask_decode": "process_mask_upsample_false_at160", "GT_decode": "overlap_IDs1_to_nl_in_existing_target_order",
                "box_coordinates": "RAW_NMS_XYXY_ON640_CANVAS_WITHOUT_EXTRA_CLIP",
                "NMS_execution": "ONE_IMAGE_PER_CALL_NATIVE_DEFAULT_TIMER_TIMEOUT_RAISES",
                "NMS_backend": "torchvision_if_available" if "torchvision" in sys.modules else "TorchNMS",
                "AP_computed": False, "new_model_forwards": 0, "new_backward": 0, "optimizer_updates": 0,
                "formal_VAL_passes": 0, "TEST_access": 0}}
    return report
