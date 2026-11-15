"""Evaluate both predeclared final A11 master states on frozen VAL after separate approval."""

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.exp13_cloud.g3.g3_batch_executor import write_json_exclusive
from tools.exp14.audit_admission import EXPECTED_MANIFEST_SHA256, read_rows, require, sha256_file
from tools.exp14.label_plan import CLASSES


STANDARD_CONF, FIXED_CONF, NMS_IOU, MATCH_IOU = 0.001, 0.25, 0.70, 0.50
SIZE_THRESHOLDS = (0.0020196759259259256, 0.007364969135802469, 0.029209280303030303)
SIZE_NAMES = ("tiny", "small", "medium", "large")
VAL_IMAGES, VAL_INSTANCES, BATCH, IMAGE_SIZE, MAX_DET = 154, 296, 32, 640, 300
EXPECTED_UPDATES = {"e0": 2100, "e1": 3300}
GATE_SCHEMA = "exp14_a11_full_val_gate_v1"
EVALUATION = {"split": "val", "checkpoint": "epoch100_last_master_fp32", "imgsz": IMAGE_SIZE,
              "conf": STANDARD_CONF, "nms_iou": NMS_IOU, "max_det": MAX_DET, "fixed_conf": FIXED_CONF,
              "match_iou": MATCH_IOU, "size_thresholds": list(SIZE_THRESHOLDS),
              "gt_size_definition": "native_rasterized_polygon_area_over_full_image_area",
              "matching": "NATIVE_ULTRALYTICS_CLASS_AWARE_IOU_GREEDY_ONE_TO_ONE",
              "test_model_inference": False}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_gate(gate, protocol_sha, identity_sha, checkpoint_sha):
    expected = {"schema": GATE_SCHEMA, "authorized": True, "protocol_sha256": protocol_sha,
                "identity_sha256": identity_sha, "checkpoint_sha256": checkpoint_sha}
    require(gate == expected and gate["authorized"] is True,
            "VAL-only Gate must bind this protocol, run identity and both epoch-100 last.pt hashes")


def size_bin(area):
    """The Exp02.2 TRAIN-derived, mutually exclusive relative mask-area bins."""
    require(math.isfinite(area) and 0 < area <= 1, "Invalid original-size GT area")
    return SIZE_NAMES[sum(area > boundary for boundary in SIZE_THRESHOLDS)]


def native_match_pairs(pred_classes, true_classes, iou):
    """Preserve GT identities in the pinned Ultralytics one-to-one IoU=.50 match."""
    require(pred_classes.ndim == true_classes.ndim == 1
            and iou.shape == (len(true_classes), len(pred_classes)) and np.isfinite(iou).all(),
            "Invalid fixed-confidence pairwise IoU array")
    require(all(np.all((ids >= 0) & (ids < len(CLASSES)) & (ids == np.floor(ids)))
                for ids in (pred_classes, true_classes)), "Prediction/GT class differs from frozen mapping")
    iou = iou * (true_classes[:, None] == pred_classes)
    pairs = np.array(np.nonzero(iou >= MATCH_IOU)).T
    if len(pairs) > 1:
        pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs


class SizeStats:
    def __init__(self):
        self.counts = defaultdict(Counter)

    def add(self, gt_classes, areas, pairs):
        require(len(gt_classes) == len(areas), "GT size and class arrays differ")
        matched = {int(i) for i in pairs[:, 0]}
        require(len(matched) == len(pairs) and all(0 <= i < len(areas) for i in matched),
                "Size audit received invalid target matches")
        for i, (cls, area) in enumerate(zip(gt_classes, areas, strict=True)):
            self.counts[int(cls), size_bin(area)]["gt"] += 1
            self.counts[int(cls), size_bin(area)]["mask_tp"] += i in matched

    def report(self):
        def result(counts):
            gt, tp = counts["gt"], counts["mask_tp"]
            return {"gt": gt, "mask_tp": tp, "mask_fn": gt - tp, "mask_recall": tp / gt if gt else None}

        return {"overall": {size: result(sum((self.counts[i, size] for i in range(len(CLASSES))), Counter()))
                            for size in SIZE_NAMES},
                "per_class": {name: {size: result(self.counts[i, size]) for size in SIZE_NAMES}
                              for i, name in enumerate(CLASSES)}}


def polygon_metadata(path, width, height):
    """Use Exp02.2's original-resolution OpenCV rasterization for the area strata."""
    import cv2

    polygons, classes, areas = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        require(len(fields) >= 7 and len(fields) % 2 == 1, "Invalid frozen segmentation label")
        cls = int(fields[0])
        polygon = np.asarray([float(v) for v in fields[1:]], dtype=np.float32).reshape(-1, 2)
        require(0 <= cls < len(CLASSES) and np.all((polygon >= 0) & (polygon <= 1)),
                "Frozen VAL polygon or class outside normalized bounds")
        points = np.rint(polygon * np.asarray([width - 1, height - 1])).astype(np.int32)
        raster = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(raster, [points], 1)
        area = float(raster.sum() / (width * height))
        size_bin(area)
        polygons.append(polygon)
        classes.append(cls)
        areas.append(area)
    return {"polygons": polygons, "classes": np.array(classes, dtype=int), "areas": np.array(areas)}


def val_source(root, row, kind, raw_image_root):
    require(kind in {"image", "label"} and row["split"] == "val", "Only VAL file paths are permitted")
    stem, suffix = row["stem"], row["image_suffix"]
    require(stem not in {"", ".", ".."} and Path(stem).name == stem and suffix in {".jpg", ".jpeg", ".png"},
            "Invalid VAL name")
    path = root / ("images" if kind == "image" else "labels") / "val" / (stem + (suffix if kind == "image" else ".txt"))
    require(path.parent.resolve(strict=True) == path.parent, "VAL directory is not a direct frozen directory")
    source = path.resolve(strict=True)
    raw = Path(row["image_path"])
    linked_image = (kind == "image" and path.is_symlink() and raw_image_root is not None
                    and not raw.is_absolute() and ".." not in raw.parts and raw.name == path.name
                    and source == raw_image_root / raw)
    key = "sha256" if kind == "image" else "label_sha256"
    require((source == path or linked_image) and source.is_file() and sha256_file(source) == row[key],
            "Frozen VAL file differs from declared identity")
    return path, source, row[key]


def stage_frozen_val(val_root, output, raw_image_root):
    sheet = val_root / "split_manifest.csv"
    require(sha256_file(sheet) == EXPECTED_MANIFEST_SHA256, "Frozen split manifest identity differs")
    rows = [r for r in read_rows(sheet) if r["split"] == "val"]
    require(len(rows) == len({r["stem"] for r in rows}) == VAL_IMAGES
            and sum(int(r["instance_count"]) for r in rows) == VAL_INSTANCES, "Unexpected frozen VAL membership")
    copy_root = output / "frozen_val"
    for key, suffix in (("images", "image_suffix"), ("labels", None)):
        source_dir = val_root / key / "val"
        require({p.name for p in source_dir.iterdir()} == {r["stem"] + (r[suffix] if suffix else ".txt") for r in rows},
                "Unexpected file member in frozen VAL")
        (copy_root / key / "val").mkdir(parents=True)
    entries, labels = [], {}
    for row in rows:
        members = {}
        for kind in ("image", "label"):
            original, source, digest = val_source(val_root, row, kind, raw_image_root)
            relative = original.relative_to(val_root)
            dst = copy_root / relative
            with source.open("rb") as reader, dst.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            require(sha256_file(dst) == digest, "Local VAL evaluation copy differs")
            if kind == "image":
                with dst.open("rb") as image:
                    if image.read(2) == b"\xff\xd8":
                        image.seek(-2, os.SEEK_END)
                        require(image.read(2) == b"\xff\xd9", "Native loader would modify this frozen JPEG")
                members["dimensions"] = (int(row["image_width"]), int(row["image_height"]))
            members[kind] = {"path": relative.as_posix(), "sha256": digest}
        image = copy_root / members["image"]["path"]
        label = copy_root / members["label"]["path"]
        item = polygon_metadata(label, *members["dimensions"])
        require(len(item["classes"]) == int(row["instance_count"]), "Frozen VAL label count differs")
        labels[image] = {**item, "dimensions": members["dimensions"]}
        entries.append(members)
    config = output / "val_only.yaml"
    write_json_exclusive(config, {"path": str(copy_root), "train": None, "val": "images/val", "names": list(CLASSES)})
    write_json_exclusive(output / "val_files.json", {"split_manifest_sha256": EXPECTED_MANIFEST_SHA256, "entries": entries})
    return config, copy_root / "images" / "val", labels, entries


def accepted_checkpoint(arm, directory, protocol, protocol_sha, identity_sha):
    receipt = load_json(directory / "summary.json")
    counts = receipt["counts"]
    updates = protocol["arms"][arm]["effective_updates"]
    pinned = ("dataset_content_sha256", "deployment_manifest_sha256", "frozen_split_sha256", "coco_weight_sha256")
    require(receipt["status"] == "PASS_FULL_TRAIN_ARM" and receipt["arm"] == arm
            and receipt["protocol_sha256"] == protocol_sha and receipt["identity_sha256"] == identity_sha
            and receipt["training_performed"] is True
            and receipt["backward_called"] is True and receipt["data_before"] == receipt["data_after"]
            and all(receipt["data_after"][key] == protocol["identities"][key] for key in pinned)
            and counts["complete_epochs"] == receipt["checkpoint"]["completed_epochs"] == 100
            and receipt["checkpoint"]["strict_terminal_reload"] is True
            and updates == EXPECTED_UPDATES[arm] == receipt["checkpoint"]["optimizer_updates"]
            and all(receipt[k] == updates for k in ("optimizer_updates", "native_steps", "optimizer_attempts"))
            and all(counts[k] == updates for k in ("batches_read", "forwards", "backwards", "finite_forwards", "completed_backwards"))
            and all(counts[k] == protocol["arms"][arm][k]
                    for k in ("original_exposures", "repair_exposures", "raw_instance_exposures"))
            and all(receipt[k] == 0 for k in ("amp_skips", "val_model_inference_runs", "test_model_inference_runs")),
            "VAL requires both fully audited 100-epoch formal training arms")
    path = directory / "last.pt"
    digest = receipt["checkpoint"]["sha256"]
    require(path.is_file() and path.resolve(strict=True) == path and sha256_file(path) == digest,
            "Fixed epoch-100 last.pt identity differs")
    return path, receipt, digest


def read_final_master(path, receipt):
    from ultralytics.nn.tasks import SegmentationModel

    from tools.exp14.a11_full_numerics import read_epoch_checkpoint
    from tools.exp14.a11_smoke_numerics import hash_state, model_fp32_finite, parameters_sha

    arm = receipt["arm"]
    record = read_epoch_checkpoint(path, expected_arm=arm)
    keys = ("protocol_sha256", "identity_sha256", "deployment_manifest_sha256", "source_weight_sha256")
    require(record["metadata"]["completed_epochs"] == 100 and record["effective_updates"] == EXPECTED_UPDATES[arm]
            and record["model_state_sha256"] == receipt["checkpoint"]["model_state_sha256"]
            and all(record["identity"][key] == receipt[key] for key in keys)
            and record["identity"]["a10_content_sha256"] == receipt["data_after"]["dataset_content_sha256"]
            and record["identity"]["split_sha256"] == EXPECTED_MANIFEST_SHA256
            and record["initial_state_sha256"] == receipt["initial_state_sha256"],
            "Formal checkpoint does not agree with the complete run receipt")
    model = SegmentationModel(record["model_config"], ch=3, nc=len(CLASSES), verbose=False)
    model.load_state_dict(record["model_state"], strict=True)
    model.names = dict(enumerate(CLASSES))
    require(model_fp32_finite(model) and parameters_sha(model) == record["parameters_sha256"]
            and hash_state(model.state_dict()) == record["model_state_sha256"], "Strict final-master reload differs")
    return model


def evaluate_master(model, config, val_dir, labels, output, confidence):
    """Run a pinned native validator after Gate, preserving GT ordinals for the fixed size report."""
    import torch
    from ultralytics.data.augment import Format
    from ultralytics.data.utils import polygons2masks_overlap
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from ultralytics.utils.metrics import mask_iou

    from tools.exp14.evaluate_formal_e0_e1 import fixed_counts

    class SizeTracingFormat(Format):
        def apply_instances(self, labels, params):
            # Only observe the permutation from the same downsampled overlap raster used by native Format.
            ordering = []
            if params["nl"]:
                _, ordering = polygons2masks_overlap((params["h"], params["w"]), params["instances"].segments,
                                                    downsample_ratio=self.mask_ratio)
            labels["frozen_gt_order"] = tuple(int(i) for i in ordering)
            return super().apply_instances(labels, params)

    class PairedValidator(SegmentationValidator):
        def __init__(self, **kwargs):
            self.size_stats = SizeStats()
            self.fixed = None
            self.observed = set()
            super().__init__(**kwargs)

        def run_callbacks(self, event):
            pass

        def get_dataloader(self, dataset_path, batch_size):
            require(self.args.split == "val" and Path(dataset_path).resolve() == val_dir,
                    "Only staged frozen VAL is authorized")
            return super().get_dataloader(dataset_path, batch_size)

        def build_dataset(self, img_path, mode="val", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            seen = set()
            for label in dataset.labels:
                image = Path(label["im_file"]).resolve(strict=True)
                original = labels[image]
                native_polygons, original_polygons = label["segments"], original["polygons"]
                require(image not in seen
                        and tuple(label["shape"]) == original["dimensions"][::-1]
                        and np.array_equal(label["cls"].reshape(-1), original["classes"])
                        and len(native_polygons) == len(original_polygons)
                        and all(np.array_equal(a, b) for a, b in zip(native_polygons, original_polygons, strict=True)),
                        "Native VAL loader dropped, duplicated, or changed a frozen target")
                seen.add(image)
            require(seen == labels.keys() and len(dataset) == VAL_IMAGES, "Native loader changed frozen VAL membership")
            if confidence == FIXED_CONF:
                formatter = dataset.transforms[-1]
                require(type(formatter) is Format and formatter.mask_overlap and formatter.mask_ratio == 4,
                        "Pinned native VAL mask formatter differs")
                dataset.transforms[-1] = SizeTracingFormat(**vars(formatter))
            return dataset

        def _prepare_batch(self, si, batch):
            result = super()._prepare_batch(si, batch)
            image = Path(result["im_file"]).resolve(strict=True)
            require(image not in self.observed and image in labels, "Duplicate or unexpected native VAL image")
            self.observed.add(image)
            if confidence == FIXED_CONF:
                order = np.asarray(batch["frozen_gt_order"][si], dtype=int)
                original = labels[image]
                require(np.array_equal(np.sort(order), np.arange(len(original["classes"])))
                        and np.array_equal(result["cls"].cpu().numpy(), original["classes"][order]),
                        "Native overlap-mask order differs from the frozen GT identity")
                result["original_areas"] = original["areas"][order]
            return result

        def _process_batch(self, preds, target):
            processed = super()._process_batch(preds, target)
            if confidence == FIXED_CONF:
                pred_cls, target_cls = (x["cls"].cpu().numpy() for x in (preds, target))
                iou = mask_iou(target["masks"].flatten(1), preds["masks"].flatten(1).float()).cpu().numpy() if (
                    len(pred_cls) and len(target_cls)) else np.zeros((len(target_cls), len(pred_cls)), dtype=float)
                pairs = native_match_pairs(pred_cls, target_cls, iou)
                require(np.array_equal(np.sort(pairs[:, 1]), np.flatnonzero(processed["tp_m"][:, 0])),
                        "Size audit fixed mask matches differ from the pinned validator's matches")
                self.size_stats.add(target_cls, target["original_areas"], pairs)
            return processed

        def get_stats(self):
            if confidence == FIXED_CONF:
                stats = {k: np.concatenate(v) for k, v in self.metrics.stats.items()}
                require(np.all(stats["conf"] >= FIXED_CONF), "Unexpected prediction below fixed confidence")
                self.fixed = fixed_counts(stats["pred_cls"], stats["target_cls"], stats["tp_m"][:, 0], stats["tp"][:, 0])
            return super().get_stats()

    output.mkdir()
    validator = PairedValidator(save_dir=output, args={
        "data": str(config), "task": "segment", "split": "val", "device": "0", "batch": BATCH, "workers": 2,
        "imgsz": IMAGE_SIZE, "conf": confidence, "iou": NMS_IOU, "max_det": MAX_DET,
        "plots": False, "save_json": False, "save_txt": False, "augment": False, "quantize": None,
        "compile": False, "overlap_mask": True, "mask_ratio": 4, "agnostic_nms": False, "single_cls": False,
    })
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with patch("ultralytics.engine.validator.callbacks.add_integration_callbacks"), patch(
        "ultralytics.data.utils.check_font", return_value=None
    ):
        metrics = {key: float(value) for key, value in validator(model=model).items()}
    nt = validator.metrics.nt_per_class
    require(validator.seen == VAL_IMAGES and validator.observed == labels.keys()
            and int(nt.sum()) == VAL_INSTANCES and validator.args.quantize is None
            and all(math.isfinite(value) for value in metrics.values()), "Native VAL results violate frozen evaluation")

    result = {"images": validator.seen, "instances": int(nt.sum()), "overall": metrics}
    if confidence == STANDARD_CONF:
        indexes = list(validator.metrics.seg.ap_class_index)
        require(indexes == list(validator.metrics.box.ap_class_index), "Mask/box metric class identity differs")
        per_class = []
        for i, label in enumerate(CLASSES):
            if i in indexes:
                mask = validator.metrics.seg.class_result(indexes.index(i))
                box = validator.metrics.box.class_result(indexes.index(i))
            else:
                # Native AP returns no class entries when the entire prediction set is empty.
                require(not indexes or not nt[i], "A present VAL class is missing native AP")
                mask = box = (0.0,) * 4
            names = ("precision", "recall", "ap50", "ap50_95")
            row = {"class_id": i, "class": label, "gt_instances": int(nt[i])}
            for key, values in (("mask", mask), ("box", box)):
                for name, value in zip(names, values, strict=True):
                    require(math.isfinite(float(value)), "Non-finite native per-class metric")
                    row[f"{key}_{name}"] = float(value)
            per_class.append(row)
        result["per_class"] = per_class
    else:
        sizes = validator.size_stats.report()
        fixed = validator.fixed
        require(fixed is not None and fixed["overall"]["gt"] == VAL_INSTANCES
                and sum(v["gt"] for v in sizes["overall"].values()) == VAL_INSTANCES
                and sum(v["mask_tp"] for v in sizes["overall"].values()) == fixed["overall"]["mask_tp"],
                "VAL size strata do not reconcile with fixed native mask totals")
        result.update(fixed_counts=fixed, size_recall=sizes)
    write_json_exclusive(output / "evaluation.json", result)
    validator.dataloader.close()
    return result


def differences(left, right):
    require(left.keys() == right.keys(), "Evaluation arms returned different metric keys")
    return {k: (right[k] - left[k] if left[k] is not None and right[k] is not None else None) for k in left}


def run(args):
    from tools.exp14.a11_full_protocol import load_protocol, validate_identity

    protocol_path = args.protocol.resolve(strict=True)
    identity_path = args.identity.resolve(strict=True)
    protocol = load_protocol(protocol_path)
    validate_identity(identity_path, protocol_path)
    require(protocol["evaluation"] == EVALUATION, "VAL evaluator settings differ from the preregistration")
    protocol_sha, identity_sha = sha256_file(protocol_path), sha256_file(identity_path)
    runs = {arm: directory.resolve(strict=True) for arm, directory in (("e0", args.e0_run), ("e1", args.e1_run))}
    require(runs["e0"] != runs["e1"], "E0 and E1 must have independent audited run directories")
    artifacts = {arm: accepted_checkpoint(arm, folder, protocol, protocol_sha, identity_sha) for arm, folder in runs.items()}
    checkpoints = {arm: artifact[2] for arm, artifact in artifacts.items()}
    receipts = [artifacts[arm][1] for arm in EXPECTED_UPDATES]
    require(all(receipts[0][key] == receipts[1][key] and receipts[0][key] for key in (
        "initial_state_sha256", "initial_parameters_sha256", "deployment_manifest_sha256", "source_weight_sha256",
    )), "Accepted formal arms do not share the frozen initialization and data lineage")
    gate = args.apply_gate.resolve(strict=True)
    validate_gate(load_json(gate), protocol_sha, identity_sha, checkpoints)
    gate_sha = sha256_file(gate)

    output = args.output.resolve()
    val_root = args.val_root.resolve(strict=True)
    raw_root = args.raw_image_root.resolve(strict=True) if args.raw_image_root else None
    immutable_roots = [val_root, *runs.values()]
    if raw_root:
        immutable_roots.append(raw_root)
    require(not any(output.is_relative_to(root) for root in immutable_roots),
            "VAL output must be independent of both formal run directories and frozen data")
    output.mkdir()
    settings = output / "settings"
    os.environ.update(YOLO_CONFIG_DIR=str(settings), MPLCONFIGDIR=str(settings),
                      YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    import ultralytics

    require(ultralytics.__version__ == "8.4.117", "Wrong Ultralytics version for preregistered validation")
    for path, receipt, digest in artifacts.values():
        require(sha256_file(path) == digest, "Checkpoint changed after Gate")
        read_final_master(path, receipt)

    config, val_dir, labels, entries = stage_frozen_val(val_root, output, raw_root)
    write_json_exclusive(output / "evaluation_protocol.json", {
        **EVALUATION, "inference_precision": "FP32_TF32_DISABLED", "batch": BATCH, "native_mask_ratio": 4,
        "native_overlap_mask": True, "fixed_matching": "Ultralytics 8.4.117 same-class one-to-one mask/box IoU",
        "protocol_sha256": protocol_sha, "identity_sha256": identity_sha, "authorization_sha256": gate_sha,
        "checkpoint_sha256": checkpoints, "evaluator_sha256": sha256_file(Path(__file__)),
        "val_files_sha256": sha256_file(output / "val_files.json"),
        "split_manifest_sha256": EXPECTED_MANIFEST_SHA256, "val_images": VAL_IMAGES, "val_instances": VAL_INSTANCES,
    })
    results = {}
    for arm, (path, receipt, digest) in artifacts.items():
        results[arm] = {}
        for name, confidence in (("standard", STANDARD_CONF), ("fixed", FIXED_CONF)):
            require(sha256_file(path) == digest and sha256_file(gate) == gate_sha, "Evaluation authority or state changed")
            model = read_final_master(path, receipt)
            results[arm][name] = evaluate_master(model, config, val_dir, labels, output / f"{arm}_{name}", confidence)

    for entry in entries:
        for kind in ("image", "label"):
            item = entry[kind]
            require(sha256_file(output / "frozen_val" / item["path"]) == item["sha256"]
                    and (val_root / item["path"]).is_file()
                    and sha256_file(val_root / item["path"]) == item["sha256"], "Frozen VAL changed during evaluation")
    require(all(sha256_file(path) == digest for path, _, digest in artifacts.values()), "Formal checkpoint changed")
    e0, e1 = results["e0"], results["e1"]
    summary = {
        "status": "PASS_PAIRED_FROZEN_VAL", "split": "val", "primary_metric": "Mask mAP50-95",
        "protocol_sha256": protocol_sha, "identity_sha256": identity_sha, "checkpoint_sha256": checkpoints,
        "results": results,
        "standard_e1_minus_e0": differences(e0["standard"]["overall"], e1["standard"]["overall"]),
        "fixed_e1_minus_e0": differences(e0["fixed"]["fixed_counts"]["overall"], e1["fixed"]["fixed_counts"]["overall"]),
        "fixed_size_recall_e1_minus_e0": differences(
            {k: v["mask_recall"] for k, v in e0["fixed"]["size_recall"]["overall"].items()},
            {k: v["mask_recall"] for k, v in e1["fixed"]["size_recall"]["overall"].items()}),
        "val_passes": 4, "test_image_reads": 0, "test_label_reads": 0, "test_model_inference_runs": 0,
    }
    write_json_exclusive(output / "paired_val_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("protocol", "identity", "e0-run", "e1-run", "val-root", "output", "apply-gate"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--raw-image-root", type=Path, help="frozen original root if the VAL images are symlinks")
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
