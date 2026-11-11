"""Evaluate only the accepted A4 final checkpoints on frozen VAL."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import ultralytics
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.nn.tasks import SegmentationModel

from tools.exp14.audit_admission import EXPECTED_MANIFEST_SHA256, read_rows, require, sha256_file
from tools.exp14.formal_e0_e1 import BASE_GIT, validate_checkpoint
from tools.exp14.label_plan import CLASSES
from tools.exp14.smoke_trainer import write_json


STANDARD_CONF = 0.001
FIXED_CONF = 0.25
NMS_IOU = 0.70


def copy_frozen_val(data, payload, output):
    manifest = data / "split_manifest.csv"
    require(sha256_file(manifest) == EXPECTED_MANIFEST_SHA256, "Frozen split manifest differs")
    rows = [r for r in read_rows(manifest) if r["split"] == "val"]
    require(len(rows) == len({r["stem"] for r in rows}) == 154
            and sum(int(r["instance_count"]) for r in rows) == 296, "Unexpected VAL set")
    val = output / "frozen_val"
    val.mkdir()
    entries = []
    for folder in ("images", "labels"):
        (val / folder / "val").mkdir(parents=True)
    for row in rows:
        stem = row["stem"]
        require(Path(stem).name == stem and row["image_suffix"] in {".jpg", ".jpeg", ".png"}, "Invalid VAL path")
        image = f"images/val/{stem}{row['image_suffix']}"
        label = f"labels/val/{stem}.txt"
        for relative, sha in ((image, row["sha256"]), (label, row["label_sha256"])):
            source = (data / relative).resolve(strict=True)
            require(source.is_relative_to(data) and source.parent.name == "val"
                    and sha256_file(source) == sha, "Frozen VAL file identity differs")
            with source.open("rb") as source_file, (val / relative).open("xb") as destination:
                shutil.copyfileobj(source_file, destination)
            require(sha256_file(val / relative) == sha, "VAL copied file differs")
        entries.append({"image": image, "label": label, "image_sha256": row["sha256"],
                        "label_sha256": row["label_sha256"], "instances": int(row["instance_count"])})
    manifest_out = output / "val_files.json"
    write_json(manifest_out, {"split_manifest_sha256": EXPECTED_MANIFEST_SHA256, "entries": entries})
    config = output / "val_only.yaml"
    write_json(config, {"path": str(val), "train": str(payload / "dataset/e0_unique_images.txt"),
                        "val": "images/val", "names": list(CLASSES)})
    return config, val / "images/val", sha256_file(manifest_out)


def fixed_counts(pred_cls, target_cls, mask_tp, box_tp):
    require(pred_cls.ndim == target_cls.ndim == mask_tp.ndim == box_tp.ndim == 1
            and len(pred_cls) == len(mask_tp) == len(box_tp)
            and np.all((pred_cls >= 0) & (pred_cls < len(CLASSES)))
            and np.all((target_cls >= 0) & (target_cls < len(CLASSES))), "Invalid fixed-threshold stats")
    per_class = []
    for i, name in enumerate(CLASSES):
        predictions = pred_cls == i
        gt = int((target_cls == i).sum())
        count = int(predictions.sum())
        mask = int((predictions & mask_tp).sum())
        box = int((predictions & box_tp).sum())
        require(mask <= gt and box <= gt, "More matched detections than targets")
        per_class.append({"class_id": i, "class": name, "gt": gt, "detections": count,
                          "mask_tp": mask, "mask_fp": count - mask, "mask_fn": gt - mask,
                          "box_tp": box, "box_fp": count - box, "box_fn": gt - box})
    totals = {k: sum(row[k] for row in per_class)
              for k in ("gt", "detections", "mask_tp", "mask_fp", "mask_fn", "box_tp", "box_fp", "box_fn")}
    totals["mask_precision"] = totals["mask_tp"] / totals["detections"] if totals["detections"] else 0.0
    totals["mask_recall"] = totals["mask_tp"] / totals["gt"] if totals["gt"] else 0.0
    return {"overall": totals, "per_class": per_class}


class FrozenValEvaluator(SegmentationValidator):
    def __init__(self, val_dir, *args, **kwargs):
        self.val_dir = val_dir
        self.fixed_stats = None
        super().__init__(*args, **kwargs)

    def get_dataloader(self, dataset_path, batch_size):
        require(Path(dataset_path).resolve() == self.val_dir, "Evaluation path outside frozen VAL")
        return super().get_dataloader(dataset_path, batch_size)

    def run_callbacks(self, event):
        # Prevent external integrations from consuming private validation results.
        pass

    def get_stats(self):
        if self.args.conf == FIXED_CONF:
            stats = self.metrics.stats
            pred = np.concatenate(stats["pred_cls"])
            target = np.concatenate(stats["target_cls"])
            require(np.all(np.concatenate(stats["conf"]) >= FIXED_CONF), "Unexpected detection below fixed confidence")
            self.fixed_stats = fixed_counts(pred, target, np.concatenate(stats["tp_m"])[:, 0],
                                           np.concatenate(stats["tp"])[:, 0])
        return super().get_stats()


def class_metrics(metrics):
    values = []
    for i, box_class in enumerate(metrics.box.ap_class_index):
        c = int(box_class)
        require(c == int(metrics.seg.ap_class_index[i]), "Mask/box category mapping differs")
        bp, br, b50, b95 = metrics.box.class_result(i)
        mp, mr, m50, m95 = metrics.seg.class_result(i)
        values.append({
            "class_id": c, "class": CLASSES[c], "gt_instances": int(metrics.nt_per_class[c]),
            "box_precision": float(bp), "box_recall": float(br), "box_ap50": float(b50), "box_ap50_95": float(b95),
            "mask_precision": float(mp), "mask_recall": float(mr), "mask_ap50": float(m50), "mask_ap50_95": float(m95),
        })
    require(len(values) == len(CLASSES) and sum(r["gt_instances"] for r in values) == 296,
            "VAL metrics did not cover expected classes and instances")
    return values


def evaluate_one(checkpoint, config, val_dir, output, conf):
    model = SegmentationModel(checkpoint["model_config"], ch=3, nc=7, verbose=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.names = dict(enumerate(CLASSES))
    validator = FrozenValEvaluator(
        val_dir, save_dir=output, args={"data": str(config), "task": "segment", "split": "val", "device": "0",
                                       "batch": 32, "workers": 2, "imgsz": 640, "conf": conf, "iou": NMS_IOU,
                                       "max_det": 300, "plots": False, "save_json": False, "save_txt": False,
                                       "augment": False, "quantize": None},
    )
    begin = time.monotonic()
    results = validator(model=model)
    require(validator.seen == 154, "Expected 154 evaluated VAL images")
    require(validator.args.quantize is None, "Evaluation precision differs from frozen FP32")
    result = {"elapsed_seconds": time.monotonic() - begin, "images": validator.seen,
              "overall": {k: float(v) for k, v in results.items()}}
    if conf == STANDARD_CONF:
        result["per_class"] = class_metrics(validator.metrics)
    else:
        require(validator.fixed_stats is not None and validator.fixed_stats["overall"]["gt"] == 296,
                "Fixed confidence evaluation missing")
        result["fixed_counts"] = validator.fixed_stats
    write_json(output / "evaluation.json", result)
    return result


def run(args):
    runtime = args.runtime.resolve(strict=True)
    payload = args.payload.resolve(strict=True)
    require(ultralytics.__version__ == "8.4.117", "Evaluator version differs")
    audit = json.loads((runtime / "training_audit.json").read_text())
    require(audit["status"] == "PASS_PAIRED_FORMAL_BUDGET" and audit["base_git"] == BASE_GIT,
            "Both arms must pass paired budget and checkpoint audit before VAL")
    checkpoint_meta = {}
    for arm in ("e0", "e1"):
        path = runtime / f"train_{arm}/last.pt"
        summary = json.loads((path.parent / "summary.json").read_text())
        sha = sha256_file(path)
        require(summary["status"] == "PASS_FORMAL_ARM"
                and sha == summary["checkpoint"]["sha256"] == audit["arms"][arm]["checkpoint"]["sha256"],
                "Checkpoint identity differs")
        validate_checkpoint(path, arm)
        checkpoint_meta[arm] = sha
    output = runtime / "val_evaluation"
    output.mkdir()
    config, val_dir, val_hash = copy_frozen_val(args.val_root.resolve(strict=True), payload, output)
    write_json(output / "evaluation_protocol.json", {
        "val_files_sha256": val_hash, "val_images": 154, "instances": 296, "split": "val",
        "evaluator_sha256": sha256_file(Path(__file__)), "ultralytics": ultralytics.__version__,
        "checkpoint_sha256": checkpoint_meta, "precision": "FP32", "batch": 32, "imgsz": 640,
        "conf": STANDARD_CONF, "nms_iou": NMS_IOU, "max_det": 300, "fixed_conf": FIXED_CONF,
        "fixed_matching": "Ultralytics native class-aware one-to-one matching at mask/box IoU >= 0.50",
    })
    results = {}
    for arm in ("e0", "e1"):
        path = runtime / f"train_{arm}/last.pt"
        require(sha256_file(path) == checkpoint_meta[arm], "Checkpoint changed before evaluation")
        checkpoint = validate_checkpoint(path, arm)
        standard = evaluate_one(checkpoint, config, val_dir, output / f"{arm}_standard", STANDARD_CONF)
        fixed = evaluate_one(checkpoint, config, val_dir, output / f"{arm}_fixed", FIXED_CONF)
        results[arm] = {"standard": standard, "fixed": fixed}
    e0, e1 = results["e0"], results["e1"]
    standard = {k: v - e0["standard"]["overall"][k] for k, v in e1["standard"]["overall"].items()}
    fixed = {k: v - e0["fixed"]["fixed_counts"]["overall"][k]
             for k, v in e1["fixed"]["fixed_counts"]["overall"].items()}
    require(all(sha256_file(runtime / f"train_{arm}/last.pt") == sha for arm, sha in checkpoint_meta.items()),
            "Checkpoint changed during evaluation")
    for row in json.loads((output / "val_files.json").read_text())["entries"]:
        for key in ("image", "label"):
            require(sha256_file(output / "frozen_val" / row[key]) == row[f"{key}_sha256"],
                    "Frozen VAL copy changed")
    summary = {"status": "PASS_VAL", "results": results, "standard_e1_minus_e0": standard,
               "fixed_e1_minus_e0": fixed, "test_reads": 0}
    write_json(output / "paired_val_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    run(parser.parse_args())
