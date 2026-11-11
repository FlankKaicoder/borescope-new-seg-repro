"""Freeze private Exp14-A2 files and an exact CPU-checked matched-slot schedule."""

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageOps

from tools.exp14.aligned_support import HOLD, READY, load_cohort, screen_core, write_json
from tools.exp14.audit_admission import EXPECTED_MANIFEST_SHA256, read_json, read_rows, require, sha256_file, under
from tools.exp14.import_human_review import trusted_path, unique_index, verified_file
from tools.exp14.label_plan import CLASSES, polygon_rows


ULTRALYTICS_VERSION = "8.4.117"
WHEEL_SHA256 = "859a16eebc707f8b1b6e259d01f70040287f2e97c12c9ee0c5e6309484db3a2f"
COCO_SHA256 = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
EPOCHS, SEED, BATCH = 100, 42, 32


def folder_under(root, value):
    directory = Path(value).resolve(strict=True)
    require(directory.is_dir() and directory.is_relative_to(root) and directory != root,
            "Directory must be inside the private root")
    return directory


def encode_labels(labels):
    return "".join(f"{row}\n" for _, row in labels.values()).encode("utf-8")


def write_csv(path, rows, fields=None):
    require(fields is not None or rows, "CSV needs fields")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields if fields is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_verified(source, target, digest):
    require(sha256_file(source) == digest, "Source file hash changed before copy")
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    require(sha256_file(target) == digest, "Copied bytes differ")


def write_bytes(path, value):
    with path.open("xb") as handle:
        handle.write(value)


def read_assessments(output, reports, provenance):
    support_file = under(output, "aligned_support_analysis.json")
    stored = read_json(support_file)
    require(json.dumps(stored, sort_keys=True) == json.dumps(
        {"provenance": provenance, "reports": reports}, sort_keys=True), "Support analysis differs on replay")
    report_sha = sha256_file(support_file)
    review_path = under(output, "full_frame_visual_assessments.csv")
    decisions = unique_index(read_rows(review_path), "variant_id")
    require(decisions.keys() == {r["variant_id"] for r in reports}, "Visual screen does not cover all core candidates")
    for record in reports:
        variant = record["variant_id"]
        decision = decisions[variant]
        proof = under(output, f"support_visuals/{variant}_support.png")
        require(decision["aligned_sha256"] == record["aligned_sha256"]
                and decision["support_report_sha256"] == report_sha
                and decision["visual_sha256"] == sha256_file(proof)
                and decision["status"] in {READY, HOLD}
                and decision["finding"] and decision["support_location"]
                and decision["assessment_origin"] == "A2_FULL_FRAME_VISUAL_COMPARISON",
                f"{variant}: visual assessment identity differs")
        require(all(region["valid_fraction"] == 1 for region in record["regions"].values()),
                f"{variant}: original target support no longer matches A1")
    return decisions, report_sha, sha256_file(review_path)


def load_ultralytics_parser(wheel):
    """Execute the pinned wheel's parser functions for NumPy/PIL inputs, without importing Torch."""
    import numpy as np

    require(sha256_file(wheel) == WHEEL_SHA256, "Ultralytics wheel does not match pinned 8.4.117")
    cpu_namespace = {"np": np, "os": os, "Image": Image, "ImageOps": ImageOps,
                     "torch": SimpleNamespace(Tensor=()), "FORMATS_HELP_MSG": "Unsupported image format"}
    functions = {
        "ultralytics/utils/ops.py": {"empty_like", "segments2boxes", "xyxy2xywh"},
        "ultralytics/data/utils.py": {"img2label_paths", "exif_size", "check_image", "verify_image_label"},
    }
    with zipfile.ZipFile(wheel) as archive:
        for module, names in functions.items():
            nodes = ast.parse(archive.read(module)).body
            selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
            if module == "ultralytics/data/utils.py":
                formats = next(node for node in nodes if isinstance(node, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == "IMG_FORMATS" for t in node.targets))
                cpu_namespace["IMG_FORMATS"] = ast.literal_eval(formats.value)
            require({node.name for node in selected} == names, "Expected pinned parser functions missing")
            exec(compile(ast.Module(body=selected, type_ignores=[]), module, "exec"), cpu_namespace)
    return cpu_namespace["img2label_paths"], cpu_namespace["verify_image_label"], np.__version__


def baseline_rows(train_root, train, dataset):
    split_path = under(train_root, "frozen_split_manifest.csv")
    frozen = unique_index([row for row in read_rows(split_path) if row["split"] == "train"], "stem")
    require(sha256_file(split_path) == EXPECTED_MANIFEST_SHA256 and frozen.keys() == train.keys(),
            "Frozen TRAIN baseline changed")
    result = {}
    for source_id, row in sorted(train.items()):
        identity = frozen[source_id]
        source = under(train_root, row["export_image_path"])
        annotation = under(train_root, row["export_json_path"])
        doc = read_json(annotation)
        original_label = polygon_rows(source_id, doc["shapes"], (int(row["image_width"]), int(row["image_height"])))
        label_bytes = encode_labels(original_label)
        label_hash = hashlib.sha256(label_bytes).hexdigest()
        require(label_hash == identity["label_sha256"] and len(original_label) == int(identity["instance_count"]),
                f"{source_id}: reconstructed labels differ from frozen original")
        image_path = Path("images/base") / f"{source_id}{source.suffix.lower()}"
        label_path = Path("labels/base") / f"{source_id}.txt"
        copy_verified(source, dataset / image_path, row["export_image_sha256"])
        write_bytes(dataset / label_path, label_bytes)
        result[source_id] = {
            "source_id": source_id, "group_id": row["group_id"], "split": "train",
            "image_path": image_path.as_posix(), "label_path": label_path.as_posix(),
            "image_sha256": row["export_image_sha256"], "label_sha256": label_hash,
            "instance_count": len(original_label), "source_image_path": str(source),
            "annotation_path": str(annotation), "annotation_sha256": row["export_json_sha256"],
        }
    require(len(result) == 668 and sum(r["instance_count"] for r in result.values()) == 1266,
            "Frozen baseline counts differ")
    return result


def materialize_core(items, decisions, dataset, root):
    frozen, regions, derived = [], [], {}
    for item in items:
        row, image = item["candidate"], item["image"]
        vid, sid = row["variant_id"], row["source_id"]
        assessment = decisions[vid]
        record = {**row, "training_status": assessment["status"], "support_location": assessment["support_location"],
                  "ui_flag": assessment["ui_flag"], "finding": assessment["finding"],
                  "a1_human_a": "PASS", "a1_human_b": "PASS",
                  "geometry_diagnostic_path": str(item["diagnostic"]),
                  "geometry_diagnostic_sha256": sha256_file(item["diagnostic"]),
                  "derived_image_path": "", "derived_label_path": "", "derived_label_sha256": ""}
        if assessment["status"] == READY:
            image_rel = Path("images/aligned") / f"{vid}.png"
            label_rel = Path("labels/aligned") / f"{vid}.txt"
            require(image.path == verified_file(root, row["final_image_path"], image.sha256)
                    and image.path.name == "aligned_output.png", "Approved aligned identity differs")
            copy_verified(image.path, dataset / image_rel, image.sha256)
            write_bytes(dataset / label_rel, b"")
            label_sha = sha256_file(dataset / label_rel)
            record.update(derived_image_path=image_rel.as_posix(), derived_label_path=label_rel.as_posix(),
                          derived_label_sha256=label_sha)
            derived[sid] = {"image_path": image_rel.as_posix(), "label_path": label_rel.as_posix(),
                            "image_sha256": image.sha256, "label_sha256": label_sha, "instance_count": 0}
        frozen.append(record)
        for polygon_id in item["region_ids"]:
            class_id = item["labels"][polygon_id][0]
            regions.append({
                **{k: record[k] for k in (
                    "variant_id", "source_id", "group_id", "source_image_path", "source_image_sha256",
                    "annotation_path", "annotation_sha256", "final_image_path", "final_image_sha256",
                    "valid_support_mask_path", "valid_support_mask_sha256", "geometry_diagnostic_path",
                    "geometry_diagnostic_sha256", "review_sha256", "bundle_id", "a1_human_a",
                    "a1_human_b", "training_status", "derived_image_path")},
                "polygon_id": polygon_id, "class_id": class_id, "class": CLASSES[class_id],
                "source_polygon_xy": item["shapes"][polygon_id]["points"],
                "aligned_region_mapping": "IDENTITY_ON_ALIGNED_ORIGINAL_SOURCE_PIXEL_GRID",
                "source_size_wh": row["final_image_size_wh"], "human_label_decision": "DELETE",
            })
    require(len(regions) == 24, "Historical core region identity count differs")
    return frozen, regions, derived


def build_slots(baseline, derived):
    require(derived.keys() <= baseline.keys(), "An extra slot is not from baseline TRAIN")
    slots = []
    for source_id, original in baseline.items():
        slots.append(slot_row(f"base:{source_id}", "BASE", original, original))
    for source_id, repaired in derived.items():
        original = baseline[source_id]
        slots.append(slot_row(f"extra:{source_id}", "EXTRA", original, repaired))
    return slots


def slot_row(slot_id, kind, original, treatment):
    row = {"slot_id": slot_id, "slot_kind": kind, "source_id": original["source_id"],
           "group_id": original["group_id"], "split": original["split"]}
    fields = ("image_path", "label_path", "image_sha256", "label_sha256", "instance_count")
    row.update({f"{arm}_{k}": item[k] for arm, item in (("e0", original), ("e1", treatment)) for k in fields})
    return row


def make_schedule(slots, epochs=EPOCHS, seed=SEED, batch=BATCH):
    require(epochs > 0 and batch > 0, "Epoch count and batch must be positive")
    indexed = unique_index(slots, "slot_id")
    require(bool(indexed), "Empty slot schedule")
    schedule = []
    for epoch in range(1, epochs + 1):
        order = sorted(indexed, key=lambda key: hashlib.sha256(f"{seed}:{epoch}:{key}".encode()).digest())
        batches = [order[i:i+batch] for i in range(0, len(order), batch)]
        schedule.append({"epoch": epoch, "batches": batches})
    return schedule


class FrozenSlotBatchSampler:
    """Map a saved matched schedule to indices in a deduplicated per-arm dataset."""

    def __init__(self, slots, schedule, dataset_paths, arm):
        require(arm in {"e0", "e1"}, "Unknown matched arm")
        require(len(dataset_paths) == len(set(dataset_paths)), "Dataset indices repeat an image path")
        indices = {p: i for i, p in enumerate(dataset_paths)}
        require(set(indices) == {r[f"{arm}_image_path"] for r in slots}, "Dataset image set differs from slots")
        self.index_by_slot = {r["slot_id"]: indices[r[f"{arm}_image_path"]] for r in slots}
        require(len(self.index_by_slot) == len(slots), "Duplicate slot")
        self.schedule = {epoch["epoch"]: epoch["batches"] for epoch in schedule}
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.batches = self.schedule[epoch]
        keys = [key for batch in self.batches for key in batch]
        require(len(keys) == len(self.index_by_slot) and set(keys) == self.index_by_slot.keys(),
                "Epoch slot set is incomplete or repeated")

    def __iter__(self):
        return ([self.index_by_slot[key] for key in batch] for batch in self.batches)

    def __len__(self):
        return len(self.batches)


def validate_schedules(slots, schedule, baseline, derived, exposure_path):
    slot_index = unique_index(slots, "slot_id")
    arm_paths = {arm: sorted({r[f"{arm}_image_path"] for r in slots}) for arm in ("e0", "e1")}
    samplers = {arm: FrozenSlotBatchSampler(slots, schedule, arm_paths[arm], arm) for arm in arm_paths}
    updates, total_slots = 0, 0
    with exposure_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "source_id", "group_id", "e0_original", "e0_aligned", "e1_original", "e1_aligned"))
        for epoch in schedule:
            batches = epoch["batches"]
            require(all(len(batch) == BATCH for batch in batches[:-1]) and 0 < len(batches[-1]) <= BATCH,
                    "Batch shape or tail differs from frozen budget")
            counts = Counter()
            for arm, sampler in samplers.items():
                sampler.set_epoch(epoch["epoch"])
                for keys, indices in zip(batches, sampler, strict=True):
                    for key, index in zip(keys, indices, strict=True):
                        slot = slot_index[key]
                        require(arm_paths[arm][index] == slot[f"{arm}_image_path"] and slot["split"] == "train",
                                "Sampler mapping or split mismatch")
                        kind = "aligned" if arm == "e1" and slot["slot_kind"] == "EXTRA" else "original"
                        counts[(arm, slot["source_id"], kind)] += 1
            for sid, row in baseline.items():
                actual = tuple(counts[(arm, sid, kind)] for arm in ("e0", "e1") for kind in ("original", "aligned"))
                extra = int(sid in derived)
                require(actual == (1 + extra, 0, 1, extra), "Per-source exposure differs")
                writer.writerow((epoch["epoch"], sid, row["group_id"], *actual))
            updates += len(batches)
            total_slots += sum(map(len, batches))
    return arm_paths, updates, total_slots


def verify_materialized(dataset, rows, img2label_paths, parser):
    images = set()
    labels = set()
    counts = Counter()
    for row in rows:
        image = under(dataset, row["image_path"])
        label = under(dataset, row["label_path"])
        require(image not in images and label not in labels, "Duplicate physical image or label")
        require(img2label_paths([str(image)]) == [str(label)], "Pinned Ultralytics image/label mapping differs")
        require(sha256_file(image) == row["image_sha256"] and sha256_file(label) == row["label_sha256"],
                "Materialized image or label bytes differ")
        if image.suffix.lower() in {".jpg", ".jpeg"}:
            with image.open("rb") as handle:
                handle.seek(-2, 2)
                require(handle.read(2) == b"\xff\xd9", "Pinned parser would rewrite a corrupt JPEG")
        checked = parser((str(image), str(label), "", False, len(CLASSES), 0, 2, False))
        _, boxes, _, segments, _, missing, found, empty, corrupt, _ = checked
        expected = row["instance_count"]
        require(not missing and found == 1 and not corrupt and empty == (expected == 0)
                and boxes.shape == (expected, 5) and len(segments) == expected,
                "Pinned Ultralytics parser did not retain the expected segmentation instances")
        require(sha256_file(image) == row["image_sha256"] and sha256_file(label) == row["label_sha256"],
                "Pinned parser changed a frozen file")
        images.add(image)
        labels.add(label)
        counts["empty_labels" if empty else "positive_labels"] += 1
        counts["segmentation_instances"] += expected
    require(images == {p for p in (dataset / "images").rglob("*") if p.is_file()}
            and labels == {p for p in (dataset / "labels").rglob("*") if p.is_file()},
            "Materialized image/label file sets differ")
    return dict(counts)


def training_protocol(frozen_split_sha, updates):
    return {
        "authorized_state": "FROZEN_DATA_PENDING_CHATGPT_SMOKE_GATE",
        "seed": SEED, "model": "yolo11n-seg.pt", "model_sha256": COCO_SHA256,
        "ultralytics": ULTRALYTICS_VERSION, "frozen_split_sha256": frozen_split_sha,
        "task": "segment", "imgsz": 640, "epochs": EPOCHS, "optimizer": "AdamW",
        "batch": BATCH, "nbs": BATCH, "accumulate": 1, "drop_last": False,
        "optimizer_updates_each_arm": updates, "deterministic": True, "amp": True,
        "lr0": 0.01, "lrf": 0.01, "cos_lr": False, "warmup_epochs": 3.0,
        "warmup_momentum": 0.8, "warmup_bias_lr": 0.1, "momentum": 0.937, "weight_decay": 0.0005,
        "mosaic": 0.0, "close_mosaic": 0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "rect": False, "multi_scale": 0.0, "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "bgr": 0.0,
        "single_image_randomness": "same seed/workers and epoch-slot order; smoke verifies observed draw ledger",
        "run_full_budget": True, "selection": "last checkpoint at frozen optimizer update",
        "validation": {"split": "val", "primary": "Mask mAP50-95", "imgsz": 640,
                       "conf": 0.001, "iou": 0.70, "test_for_method_selection": False},
    }


def freeze(root, a1, output, wheel):
    items, train_root, train, train_report, excluded, provenance = load_cohort(root, a1)
    reports = screen_core(root, items)
    decisions, support_sha, assessment_sha = read_assessments(output, reports, provenance)
    img2label, parser, numpy_version = load_ultralytics_parser(wheel)
    dataset = output / "dataset"
    dataset.mkdir()
    for path in ("images/base", "labels/base", "images/aligned", "labels/aligned"):
        (dataset / path).mkdir(parents=True)
    baseline = baseline_rows(train_root, train, dataset)
    frozen, regions, derived = materialize_core(items, decisions, dataset, root)
    require(bool(derived), "No candidate survived full-frame screening")
    slots = build_slots(baseline, derived)
    schedule = make_schedule(slots)
    arm_paths, updates, total_slots = validate_schedules(slots, schedule, baseline, derived, output / "per_source_exposure.csv")
    parser_counts = verify_materialized(dataset, [*baseline.values(), *derived.values()], img2label, parser)
    require(parser_counts == {"positive_labels": 668, "segmentation_instances": 1266, "empty_labels": len(derived)},
            "Pinned parser counts differ")
    for arm, paths in arm_paths.items():
        write_bytes(dataset / f"{arm}_unique_images.txt", "".join(f"./{path}\n" for path in paths).encode())
    write_csv(output / "original_train_manifest.csv", list(baseline.values()))
    write_csv(output / "core_freeze_manifest.csv", frozen)
    write_csv(output / "matched_slots.csv", slots)
    for row in frozen:
        if row["training_status"] == HOLD:
            excluded.append({"variant_id": row["variant_id"], "a1_a_status": "VERIFIED_CANDIDATE",
                             "a1_b_status": "VERIFIED_CANDIDATE", "reason": row["finding"]})
    write_csv(output / "not_in_training.csv", excluded)
    write_json(output / "b_region_metadata.json", regions)
    write_json(output / "epoch_slot_order.json", schedule)
    write_json(output / "training_protocol.json", training_protocol(EXPECTED_MANIFEST_SHA256, updates))
    counts = Counter(row["repair_class"] for row in frozen if row["training_status"] == READY)
    overall = [r["valid_fraction"] for r in reports]
    retained = [r["valid_fraction"] for r in reports if decisions[r["variant_id"]]["status"] == READY]
    report = {
        "stage": "Exp14-A2", "status": "FROZEN_DATA_PENDING_CHATGPT_SMOKE_GATE",
        "core_sources_screened": len(items), "core_groups_screened": len({r["group_id"] for r in frozen}),
        "retained_sources": len(derived), "training_holds": len(items) - len(derived),
        "retained_by_class": {name: counts[name] for name in CLASSES},
        "valid_frame_fraction_all_min_max": [min(overall), max(overall)],
        "valid_frame_fraction_retained_min_max": [min(retained), max(retained)],
        "target_regions_with_full_support": sum(r["valid_fraction"] == 1 for item in reports for r in item["regions"].values()),
        "ui_removal_flags_inside_valid_area": sum(r["ui_flag"] == "UI_REMOVED_INSIDE_VALID_AREA" for r in frozen),
        "b_original_regions_preserved": len(regions),
        "b_regions_on_retained_sources": sum(r["training_status"] == READY for r in regions),
        "baseline_images": len(baseline), "baseline_instances": train_report["checked_train_instances"],
        "derived_aligned_images": len(derived), "explicit_empty_labels": parser_counts["empty_labels"],
        "parser": {"ultralytics": ULTRALYTICS_VERSION, "wheel_sha256": WHEEL_SHA256, "numpy": numpy_version,
                   "pillow": Image.__version__, "scope": "PINNED_PARSER_FUNCTIONS_CPU_NUMPY_PIL",
                   **parser_counts},
        "schedule": {"epochs": EPOCHS, "batch": BATCH, "tail_batch": len(schedule[0]["batches"][-1]),
                     "steps_each_epoch": len(schedule[0]["batches"]), "optimizer_updates_each_arm": updates,
                     "base_slots_each_epoch": len(baseline), "extra_slots_each_epoch": len(derived),
                     "total_draws_each_arm": total_slots, "cross_sample_augmentation": "DISABLED"},
        "historical_trusted_pair_pass": 0,
    }
    write_json(output / "data_integrity_report.json", {
        "summary": report, "provenance": provenance, "support_report_sha256": support_sha,
        "visual_assessments_sha256": assessment_sha, "frozen_split_sha256": EXPECTED_MANIFEST_SHA256,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ultralytics-wheel", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root.resolve(strict=True)
    a1 = folder_under(root, args.a1)
    output = folder_under(root, args.output)
    require(output != a1 and output.parent == root and output.name.startswith("exp14_a2_core_freeze_"),
            "Use a dedicated Exp14-A2 output directory")
    print(json.dumps(freeze(root, a1, output, trusted_path(output, args.ultralytics_wheel)), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
