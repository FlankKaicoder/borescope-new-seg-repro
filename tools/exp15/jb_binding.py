"""Reconstruct the native overlap instance order from augmented, pre-Format Polygons."""

from collections import Counter

import cv2
import numpy as np
import torch
from ultralytics.data.augment import Format

from tools.exp14.audit_admission import require
from tools.exp15.regions import P3_CHANNELS, inspect_regions, project_regions


def reconstruct(segments, height, width, ratio):
    masks = []
    for points in segments:
        canvas = np.zeros((height, width), np.uint8)
        cv2.fillPoly(canvas, [points.astype(np.int32)], 1)
        masks.append(cv2.resize(canvas, (width // ratio, height // ratio)))
    ids = np.zeros((height // ratio, width // ratio), dtype=np.int32)
    binary = np.asarray(masks)
    order = np.argsort(-binary.sum(axis=(1, 2))) if masks else np.empty(0, int)
    # Native smaller regions are placed later and win shared pixels.
    for mask_id, original in enumerate(order, 1):
        ids[binary[original] != 0] = mask_id
    return ids, order


def layout_counts(boxes, classes):
    widths = boxes[:, 2:]
    lo, hi = boxes[:, :2] - widths / 2, boxes[:, :2] + widths / 2
    intersection = (np.minimum(hi[:, None], hi) - np.maximum(lo[:, None], lo)).clip(0).prod(axis=2)
    area = widths.prod(axis=1)
    union = area[:, None] + area - intersection
    high = np.triu(intersection >= 0.5 * union, k=1)
    return {
        "images": 1, "multi_instance_images": int(len(classes) > 1),
        "multiclass_images": int(len(set(classes)) > 1),
        "high_box_iou_pairs": int(high.sum()),
        "high_box_iou_different_class_pairs": int((high & (classes[:, None] != classes)).sum()),
        "thin_crack_instances": int(((classes == 1) & (widths.min(axis=1) <= 0.2 * widths.max(axis=1))).sum()),
        "high_box_iou": bool(high.any()),
    }


def check_formatted(expected, classes, boxes, observed):
    require(np.array_equal(observed["masks"][0].numpy(), expected)
            and np.array_equal(observed["cls"].numpy().reshape(-1), classes)
            and np.array_equal(observed["bboxes"].numpy(), boxes),
            "Native mask ID, Polygon, box and class permutation do not bind")


class AuditedFormat(Format):
    def apply_instances(self, labels, params=None):
        require(self.mask_overlap and self.return_mask and self.normalize and self.mask_ratio == 4,
                "Expected native normalized overlap Format")
        w, h = params["w"], params["h"]
        instances = params["instances"]
        ids, order = reconstruct(instances.segments, h, w, self.mask_ratio)
        classes = params["cls"].reshape(-1)[order]
        boxes = instances.bboxes[order].copy()
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        observed = super().apply_instances(labels, params)
        check_formatted(ids, classes, boxes, observed)
        observed["jb_binding"] = layout_counts(boxes, classes)
        return observed


def install_format_audit(dataset):
    transform = dataset.transforms.transforms
    require(type(transform[-1]) is Format, "Expected native Format as the final transform")
    aud = AuditedFormat()
    aud.__dict__.update(vars(transform[-1]))
    transform[-1] = aud


def check_pool_tags(batch, groups, private_dir, selected):
    """Distinguish regions with a deterministic coordinate field and a per-image tag."""
    counts = Counter()
    for slot, metadata in enumerate(batch["jb_binding"]):
        counts.update({k: v for k, v in metadata.items() if k != "high_box_iou"})
        targets = batch["batch_idx"] == slot
        single = {name: batch[name][targets] for name in ("cls", "bboxes", "batch_idx")}
        single.update(im_file=[batch["im_file"][slot]], img=batch["img"][slot:slot+1],
                      masks=batch["masks"][slot:slot+1])
        single["batch_idx"] = torch.zeros_like(single["batch_idx"])
        regions = inspect_regions(single, groups)
        counts["targets"] += len(regions.cells)
        counts["small_p3_instances"] += int(((regions.cells > 0) & (regions.cells <= 4)).sum())
        tags = {
            "high_overlap": metadata["high_box_iou"],
            "thin_crack": metadata["thin_crack_instances"] > 0,
            "small_polygon": bool(((regions.cells > 0) & (regions.cells <= 4)).any()),
            "mixed_class": metadata["multiclass_images"] > 0,
        }
        if not regions.cells.any() or not any(value and key not in selected for key, value in tags.items()):
            continue
        sy, sx = torch.meshgrid(torch.arange(80), torch.arange(80), indexing="ij")
        field = torch.zeros(1, P3_CHANNELS, 80, 80)
        field[0, 0], field[0, 1] = sy / 80, sx / 80
        field[0, 2], field[0, 3] = sy * sx / 6400, (slot + 1) / len(batch["img"])
        pooled = project_regions(field, single["masks"], regions, torch.nn.Identity())
        expected = []
        raster = single["masks"][0].numpy()
        for rank in regions.ordinals[regions.cells > 0]:
            y, x = np.nonzero(raster == rank + 1)
            expected.append(field[0].flatten(1)[:, (y // 2) * 80 + x // 2].mean(1))
        torch.testing.assert_close(pooled, torch.nn.functional.normalize(torch.stack(expected), dim=1),
                                   rtol=1e-5, atol=1e-6)
        for key, include in tags.items():
            if include and key not in selected:
                selected.add(key)
                render_sample(single, private_dir / f"{key}.png")
                counts[f"checked_tag_{key}"] += len(expected)
    return counts


def render_sample(batch, path):
    from PIL import Image, ImageDraw

    pixels = batch["img"][0].permute(1, 2, 0).numpy().copy()
    draw_labels = []
    for idx, category in enumerate(batch["cls"].reshape(-1).tolist(), 1):
        y, x = np.nonzero(batch["masks"][0].numpy() == idx)
        if len(x):
            color = np.array(((idx * 89) % 256, (idx * 173) % 256, (idx * 53) % 256), dtype=np.uint8)
            show = batch["masks"][0].numpy() == idx
            show = np.repeat(np.repeat(show, 4, axis=0), 4, axis=1)
            pixels[show] = pixels[show] // 2 + color // 2
            draw_labels.append((int(x[0]) * 4, int(y[0]) * 4, f"{idx}/{int(category)}"))
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas)
    for x, y, label in draw_labels:
        draw.text((x, y), label, fill="white", stroke_fill="black", stroke_width=2)
    canvas.save(path)
