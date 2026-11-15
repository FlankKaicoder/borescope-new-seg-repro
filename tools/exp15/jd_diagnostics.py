"""Observe the four native JD passes without changing their predictions or scoring."""

import json
from collections import Counter

import numpy as np

from tools.exp14.a11_full_eval import FIXED_CONF, MATCH_IOU, native_match_pairs
from tools.exp14.audit_admission import require

MISS = (
    "same_class_mask_edge_lost_one_to_one",
    "wrong_class_mask_overlap",
    "same_class_box_good_mask_below_50",
    "same_class_overlap_below_50",
    "no_same_class_overlap",
)
DEFINITIONS = {
    "fixed_mask_fn_priority": list(MISS),
    "mask_overlap_cutoff": MATCH_IOU,
    "nonzero_overlap_criterion": "box IoU or mask IoU strictly above zero",
    "box_match_quality": "mask IoU on the same pair selected by native fixed class-aware box matching",
    "low_conf_band": "0.001 <= score < 0.25 from standard pass post-NMS predictions",
    "low_conf_candidate_gt": "GT unmatched by native mask matching on the >=0.25 subset of the standard pass, "
                            "with >=1 same-class mask IoU >=0.50 candidate in the low band",
}


def observe(pred, gt, conf, mask, box, native, cutoff):
    m = native_match_pairs(pred, gt, mask)
    b = native_match_pairs(pred, gt, box)
    for pairs, field in ((m, "tp_m"), (b, "tp")):
        require(np.array_equal(np.sort(pairs[:, 1]), np.flatnonzero(native[field][:, 0])),
                "Diagnostic match indices do not reproduce native evaluator at IoU .50")
    counts = Counter(predictions=len(pred), gt=len(gt), native_mask_tp=len(m), native_box_tp=len(b))
    same = gt[:, None] == pred
    mask_edges = mask >= MATCH_IOU
    box_edges = box >= MATCH_IOU
    if cutoff == FIXED_CONF:
        require(np.all(conf >= FIXED_CONF), "A fixed-pass detection is below .25")
        missed = np.ones(len(gt), dtype=bool)
        missed[m[:, 0]] = False
        categories = (
            (same & mask_edges).any(axis=1),
            (~same & mask_edges).any(axis=1),
            (same & box_edges).any(axis=1),
            (same & ((box > 0) | (mask > 0))).any(axis=1),
            np.ones(len(gt), dtype=bool),
        )
        for label, category in zip(MISS, categories, strict=True):
            matched = missed & category
            counts[label] += int(matched.sum())
            missed &= ~matched
        levels = mask[b[:, 0], b[:, 1]]
        counts["box_tp_with_mask_iou_below_50"] = int((levels < MATCH_IOU).sum())
        counts["mask_iou_sum_on_box_tp"] = float(levels.astype(np.float64).sum())
    else:
        low = conf < FIXED_CONF
        high = ~low
        high_mask = native_match_pairs(pred[high], gt, mask[:, high])
        missed = np.ones(len(gt), dtype=bool)
        missed[high_mask[:, 0]] = False
        counts.update(low_band_predictions=int(low.sum()), low_band_native_mask_tp=int(native["tp_m"][low, 0].sum()),
                      low_band_native_box_tp=int(native["tp"][low, 0].sum()),
                      high_subset_mask_missed_gt=int(missed.sum()),
                      low_band_candidate_gt=int((missed & (same[:, low] & mask_edges[:, low]).any(axis=1)).sum()))
    return dict(counts)


def diagnostic_validator(native_class, trace_path, cutoff):
    import torch
    from ultralytics.utils.metrics import box_iou, mask_iou

    class DiagnosticValidator(native_class):
        def preprocess(self, batch):
            result = super().preprocess(batch)
            require(result["img"].is_cuda and result["img"].dtype == torch.float32
                    and not torch.is_autocast_enabled("cuda")
                    and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                    "Inference input is not CUDA FP32 with TF32 disabled")
            return result

        def _process_batch(self, preds, target):
            original = super()._process_batch(preds, target)
            pred, gt = (item["cls"].cpu().numpy() for item in (preds, target))
            conf = preds["conf"].cpu().numpy()
            require(conf.shape == pred.shape and np.isfinite(conf).all(), "Nonfinite or misaligned native confidences")
            if len(pred) and len(gt):
                mask = mask_iou(target["masks"].flatten(1), preds["masks"].flatten(1).float()).cpu().numpy()
                box = box_iou(target["bboxes"], preds["bboxes"]).cpu().numpy()
            else:
                mask = box = np.zeros((len(gt), len(pred)), dtype=np.float32)
            row = {"image_ordinal": self.seen, **observe(pred, gt, conf, mask, box, original, cutoff)}
            with trace_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            return original

    return DiagnosticValidator


def summarize_trace(path, metrics, cutoff):
    counts = Counter()
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) == metrics["images"] and
            [r["image_ordinal"] for r in rows] == list(range(1, len(rows) + 1)), "Incomplete audit trace")
    for row in rows:
        counts.update({k: v for k, v in row.items() if k != "image_ordinal"})
    require(counts["gt"] == metrics["instances"], "Diagnostic GT count differs")
    if cutoff == FIXED_CONF:
        fixed = metrics["fixed_counts"]["overall"]
        require(fixed["detections"] == counts["predictions"] and fixed["mask_tp"] == counts["native_mask_tp"]
                and fixed["box_tp"] == counts["native_box_tp"]
                and fixed["mask_fn"] == sum(counts[label] for label in MISS), "Diagnostic categories do not reconcile")
        total = counts.pop("mask_iou_sum_on_box_tp")
        counts["mean_mask_iou_on_box_tp"] = total / counts["native_box_tp"] if counts["native_box_tp"] else None
    return dict(counts)
