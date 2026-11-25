"""Isolated future native evaluation and lossless same-pass prediction records.

Only synthetic records are exercised in the CPU phase. No VAL data is read here
without the independent future evaluation Gate.
"""
from __future__ import annotations

import base64
import copy
import gzip
import json
import os
from pathlib import Path

from .common import digest, require, sha256, write_json_exclusive
from .protocol import require_operation


def _numpy(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def encode_masks(value):
    import numpy as np
    masks = _numpy(value)
    require(masks.ndim == 3 and np.isfinite(masks).all() and np.isin(masks, (0, 1)).all(),
            "Preserve exact native binary masks; probabilities cannot be silently thresholded")
    packed = np.packbits(masks.astype(np.uint8, copy=False).reshape(-1), bitorder="little")
    return {"encoding": "packbits_little_base64", "shape": list(masks.shape),
            "data": base64.b64encode(packed.tobytes()).decode("ascii"),
            "binary_bytes_sha256": __import__("hashlib").sha256(masks.astype(np.uint8).tobytes()).hexdigest()}


def decode_masks(record):
    import numpy as np
    require(record["encoding"] == "packbits_little_base64", "Unknown mask encoding")
    shape = tuple(record["shape"])
    require(len(shape) == 3 and all(type(n) is int and n >= 0 for n in shape), "Invalid mask shape")
    count = int(np.prod(shape))
    raw = base64.b64decode(record["data"], validate=True)
    require(len(raw) == (count + 7) // 8, "Encoded mask length differs")
    result = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little", count=count).reshape(shape)
    require(__import__("hashlib").sha256(result.tobytes()).hexdigest() == record["binary_bytes_sha256"], "Decoded mask SHA differs")
    return result


def native_match_identities(pred_classes, gt_classes, iou, thresholds):
    """Pinned native greedy ordering, including its two np.unique filters."""
    import numpy as np
    pred, gt, matrix = _numpy(pred_classes).reshape(-1), _numpy(gt_classes).reshape(-1), _numpy(iou)
    require(matrix.shape == (len(gt), len(pred)) and np.isfinite(matrix).all(), "Invalid matching matrix")
    matrix = matrix * (gt[:, None] == pred[None, :])
    correct = np.zeros((len(pred), len(thresholds)), dtype=bool)
    pairs = []
    for column, threshold in enumerate(thresholds):
        matches = np.array(np.nonzero(matrix >= threshold)).T
        if len(matches) > 1:
            matches = matches[matrix[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        if len(matches): correct[matches[:, 1], column] = True
        pairs.append(matches.tolist())
    return correct, pairs


class PredictionWriter:
    def __init__(self, directory, bindings, *, synthetic=False, protocol=None, reserve_bytes=5 * 1024**3):
        if not synthetic: require_operation("evaluate", protocol)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.bindings, self.synthetic, self.reserve_bytes = copy.deepcopy(bindings), synthetic, reserve_bytes
        self.rows, self.identities = [], set()

    def add(self, image_id, prediction, target, *, gt_ordinals, thresholds, mask_iou, box_iou, native_tp):
        import numpy as np
        require(image_id not in self.identities, "Duplicate prediction image")
        if not self.synthetic:
            import shutil
            require(shutil.disk_usage(self.directory).free >= self.reserve_bytes, "Prediction capacity reserve reached")
        pcls, gcls = _numpy(prediction["cls"]).reshape(-1), _numpy(target["cls"]).reshape(-1)
        require(sorted(gt_ordinals) == list(range(len(gcls))), "Original GT ordinal binding differs")
        pmasks, gmasks = encode_masks(prediction["masks"]), encode_masks(target["masks"])
        require(pmasks["shape"][0] == len(pcls) and gmasks["shape"][0] == len(gcls), "Mask/class count mismatch")
        tp_m, pairs_m = native_match_identities(pcls, gcls, mask_iou, thresholds)
        tp_b, pairs_b = native_match_identities(pcls, gcls, box_iou, thresholds)
        require(np.array_equal(tp_m, _numpy(native_tp["tp_m"])) and np.array_equal(tp_b, _numpy(native_tp["tp"])),
                "Saved matching identities diverge from native metrics")
        record = {"schema": "EXP18_PREDICTION_V1", "synthetic": self.synthetic, "image_id": image_id,
                  "bindings": self.bindings, "thresholds": list(thresholds), "gt_original_ordinals": list(gt_ordinals),
                  "prediction": {"cls": pcls.tolist(), "conf": _numpy(prediction["conf"]).tolist(),
                                 "bboxes": _numpy(prediction["bboxes"]).tolist(), "masks": pmasks},
                  "target": {"cls": gcls.tolist(), "bboxes": _numpy(target["bboxes"]).tolist(), "masks": gmasks},
                  "mask_iou": _numpy(mask_iou).tolist(), "box_iou": _numpy(box_iou).tolist(),
                  "mask_pairs_native_order": pairs_m, "box_pairs_native_order": pairs_b,
                  "mask_pairs_original_GT": [[[gt_ordinals[g], p] for g, p in pairs] for pairs in pairs_m],
                  "box_pairs_original_GT": [[[gt_ordinals[g], p] for g, p in pairs] for pairs in pairs_b]}
        name = f"image_{len(self.rows):05d}.json.gz"
        with (self.directory / name).open("xb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")).encode())
            raw.flush(); os.fsync(raw.fileno())
        row = {"file": name, "image_id": image_id, "sha256": sha256(self.directory / name)}
        self.rows.append(row); self.identities.add(image_id)
        return row

    def close(self, expected_images):
        require(self.identities == set(expected_images), "Prediction pass omitted an image")
        return write_json_exclusive(self.directory / "manifest.json",
            {"schema": "EXP18_PREDICTION_PASS_V1", "bindings": self.bindings, "images": self.rows,
             "synthetic": self.synthetic, "additional_inference_passes": 0})


def run_isolated_native(callback, adapter, *, epoch, role):
    require_operation("evaluate", adapter.protocol)
    return _run_isolated(callback, adapter, epoch=epoch, role=role)


def run_isolated_synthetic(callback, adapter, *, epoch, role):
    require(getattr(adapter.model, "exp18_synthetic", False) is True, "Synthetic isolation requires a marked toy graph")
    require(all(p.device.type == "cpu" for p in adapter.model.parameters()), "Synthetic isolation is CPU only")
    return _run_isolated(callback, adapter, epoch=epoch, role=role)


def _run_isolated(callback, adapter, *, epoch, role):
    require(role in ("MASTER", "EMA"), "Unknown validation role")
    import torch
    from .checkpoint import state_hash, _tree
    from tools.exp17.formal.rng import capture_rng, restore_rng
    protected = lambda: _tree({"model": adapter.model.state_dict(), "ema": adapter.ema.state_dict(),
        "optimizer": adapter.optimizer.state_dict(), "scheduler": adapter.scheduler.state_dict(),
        "scaler": adapter.scaler.state_dict() if adapter.scaler else None,
        "modes": {name: m.training for name, m in adapter.model.named_modules()},
        "requires_grad": {name: p.requires_grad for name, p in adapter.model.named_parameters()},
        "grads": {name: p.grad for name, p in adapter.model.named_parameters()},
        "counters": copy.deepcopy(getattr(adapter, "counters", {})),
        "adapter_fields": {name: copy.deepcopy(getattr(adapter, name)) for name in
                           ("current_epoch", "last_matching", "last_packet") if hasattr(adapter, name)},
        "trainer_epoch": getattr(getattr(adapter, "trainer", None), "epoch", None),
        "ema_pending": copy.deepcopy(getattr(adapter.ema, "_pending_active", None))}, native=True)
    snapshot, rng = protected(), capture_rng()
    before = state_hash(snapshot)
    flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    deterministic = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    source = adapter.model if role == "MASTER" else adapter.ema.ema
    try:
        clone = copy.deepcopy(source).float().eval()
        with torch.no_grad(): result = callback(clone, epoch=epoch, role=role)
    finally:
        restore_rng(rng)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[:2]
        torch.set_float32_matmul_precision(flags[4])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[2:4]
        torch.use_deterministic_algorithms(deterministic[0], warn_only=deterministic[1])
        changed = state_hash(protected()) != before
        if changed:
            adapter.model.load_state_dict(snapshot["model"], strict=True)
            adapter.ema.load_state_dict(snapshot["ema"])
            adapter.optimizer.load_state_dict(snapshot["optimizer"])
            adapter.scheduler.load_state_dict(snapshot["scheduler"])
            if adapter.scaler: adapter.scaler.load_state_dict(snapshot["scaler"])
            for name, module in adapter.model.named_modules(): module.training = snapshot["modes"][name]
            for name, parameter in adapter.model.named_parameters():
                parameter.requires_grad_(snapshot["requires_grad"][name])
                saved = snapshot["grads"][name]
                parameter.grad = None if saved is None else saved.to(parameter.device).clone()
            if hasattr(adapter, "counters"):
                adapter.counters.clear(); adapter.counters.update(snapshot["counters"])
            for name, value in snapshot["adapter_fields"].items(): setattr(adapter, name, value)
            if getattr(adapter, "trainer", None) is not None: adapter.trainer.epoch = snapshot["trainer_epoch"]
            if hasattr(adapter.ema, "_pending_active"): adapter.ema._pending_active = snapshot["ema_pending"]
            require(state_hash(protected()) == before, "Protected state could not be restored")
            raise ValueError("Isolated validation mutated protected training state; restored and HOLD")
    return result


def make_observed_validator(protocol, writer, *, expected_labels, base_class=None):
    """Factory for hooks inside the planned native pass, including empty images.

    The caller stages read-only VAL and its isolated label cache, constructs the
    returned validator with the pinned settings, and invokes that one pass.
    """
    require_operation("evaluate", protocol)
    import numpy as np
    from ultralytics.data.augment import Format
    from ultralytics.data.utils import polygons2masks_overlap
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from ultralytics.utils.metrics import mask_iou, box_iou
    import ultralytics.data.dataset as dataset_module
    from unittest.mock import patch
    import uuid
    from .data import exclusive_label_cache
    base_class = base_class or SegmentationValidator

    class OrdinalFormat(Format):
        def apply_instances(self, labels, params):
            order = ()
            if params["nl"]:
                _, order = polygons2masks_overlap((params["h"], params["w"]), params["instances"].segments,
                                                  downsample_ratio=self.mask_ratio)
            labels["exp18_gt_order"] = tuple(int(i) for i in order)
            return super().apply_instances(labels, params)

    class ObservedValidator(base_class):
        def run_callbacks(self, _event): pass

        def build_dataset(self, img_path, mode="val", batch=None):
            require(mode == "val", "Observer accepts VAL only")
            cache = writer.directory / ("labels_" + uuid.uuid4().hex + ".cache")
            original = dataset_module.YOLODataset._load_or_scan_cache
            def isolated_load(dataset, _native_path, cache_hash):
                require(not cache.exists(), "VAL label cache must have a fresh name")
                loaded, reused = original(dataset, cache, cache_hash)
                require(not reused and not loaded["msgs"], "VAL scan found stale cache or source warnings")
                return loaded, reused
            with patch.object(dataset_module.YOLODataset, "_load_or_scan_cache", isolated_load), \
                 patch.object(dataset_module, "save_dataset_cache_file", exclusive_label_cache):
                dataset = super().build_dataset(img_path, mode, batch)
            found = {str(Path(row["im_file"]).resolve()) for row in dataset.labels}
            require(found == set(expected_labels), "Native VAL dataset changed frozen membership")
            formatter = dataset.transforms[-1]
            require(type(formatter) is Format and formatter.mask_overlap and formatter.mask_ratio == 4,
                    "Native VAL Format differs")
            dataset.transforms[-1] = OrdinalFormat(**vars(formatter))
            return dataset

        def _prepare_batch(self, si, batch):
            target = super()._prepare_batch(si, batch)
            image = str(Path(target["im_file"]).resolve(strict=True))
            require(image in expected_labels, "Prediction image left frozen VAL")
            order = tuple(batch["exp18_gt_order"][si])
            require(np.array_equal(_numpy(target["cls"]), np.asarray(expected_labels[image]["classes"])[list(order)]),
                    "GT ordinal/class identity differs")
            target["exp18_gt_ordinals"] = order
            self._exp18_targets[si] = target
            return target

        def _process_batch(self, prediction, target):
            processed = super()._process_batch(prediction, target)
            self._record(prediction, target, processed)
            return processed

        def _record(self, prediction, target, processed):
            nc, ng = len(prediction["cls"]), len(target["cls"])
            miou = mask_iou(target["masks"].flatten(1), prediction["masks"].flatten(1).float()) if nc and ng else np.zeros((ng, nc))
            biou = box_iou(target["bboxes"], prediction["bboxes"]) if nc and ng else np.zeros((ng, nc))
            writer.add(str(Path(target["im_file"]).resolve()), prediction, target,
                       gt_ordinals=target["exp18_gt_ordinals"], thresholds=self.iouv.cpu().tolist(),
                       mask_iou=miou, box_iou=biou, native_tp=processed)

        def update_metrics(self, predictions, batch):
            self._exp18_targets = {}
            super().update_metrics(predictions, batch)
            for si, prediction in enumerate(predictions):
                target = self._exp18_targets[si]
                image = str(Path(target["im_file"]).resolve())
                if image not in writer.identities:
                    native_prediction = self._prepare_pred(prediction, target)
                    self._record(native_prediction, target, super()._process_batch(native_prediction, target))
    return ObservedValidator
