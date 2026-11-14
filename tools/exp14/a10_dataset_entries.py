"""Publish exact original TRAIN and A10 TRAIN entry lists without running training."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from PIL import Image

from tools.exp13_cloud.g3.g3_batch_executor import write_exclusive, write_json_exclusive
from tools.exp13_cloud.g3.verify_train_export import (
    EXPECTED_TRAIN_IMAGES, EXPECTED_TRAIN_INSTANCES, load_class_mapping,
)
from tools.exp14.audit_admission import EXPECTED_MANIFEST_SHA256, read_rows, require, sha256_file, under, verify_train
from tools.exp14.freeze_core import (
    ULTRALYTICS_VERSION, WHEEL_SHA256, baseline_rows, copy_verified, load_ultralytics_parser, verify_materialized,
)
from tools.exp14.label_plan import CLASSES

EVAL_ROOT = "/root/autodl-tmp/borescope-new-seg-data/v1"
WHEEL_RELATIVE = "exp14_a2_core_freeze_20261111_xpeawdm_/cpu_parser/ultralytics-8.4.117-py3-none-any.whl"
SPLIT_SIZES = {"train": EXPECTED_TRAIN_IMAGES, "val": 154, "test": 147}
SPLIT_INSTANCES = {"train": EXPECTED_TRAIN_INSTANCES, "val": 296, "test": 285}
ARM_CONFIGS = {
    "e0": ("train_e0_manifest.txt", "e0_original_668.yaml"),
    "e1": ("train_e1_manifest.txt", "e1_original_668_plus_N.yaml"),
}
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def load_frozen_split(path):
    """Verify split identities and prevent group and exact-image leakage."""
    require(sha256_file(path) == EXPECTED_MANIFEST_SHA256, "Frozen split manifest hash differs")
    rows = read_rows(path)
    require(Counter(r["split"] for r in rows) == SPLIT_SIZES
            and len({r["stem"] for r in rows}) == len(rows), "Frozen split membership or counts differ")
    groups, hashes, instances = defaultdict(set), defaultdict(set), Counter()
    by_split = {}
    for row in sorted(rows, key=lambda r: r["stem"]):
        split = row["split"]
        groups[row["group_id"]].add(split)
        hashes[row["sha256"]].add(split)
        instances[split] += int(row["instance_count"])
        by_split.setdefault(split, {})[row["stem"]] = row
    require(all(len(s) == 1 for s in [*groups.values(), *hashes.values()])
            and instances == SPLIT_INSTANCES, "Frozen group, image hash or instance counts differ")
    return by_split


def bind_entry(output, dataset, row, original, held_out_hashes):
    """Normalize approved A10 materialized copies to the shared dataset manifest."""
    source = row["source_id"]
    require(source == original["source_id"] and row["split"] == original["split"] == "train"
            and row["group_id"] == original["group_id"], "A10 asset is outside its original frozen TRAIN group")
    image = (output / row["published_path"]).resolve(strict=True)
    label = (output / row["label_path"]).resolve(strict=True)
    require(image.is_relative_to(dataset / "images") and label.is_relative_to(dataset / "labels"),
            "Frozen A10 image or label path is outside the new dataset")
    image_rel, label_rel = [p.relative_to(dataset) for p in (image, label)]
    counterpart = (Path("labels") / image_rel.relative_to("images")).with_suffix(".txt")
    require(label_rel == counterpart and image_rel != Path(original["image_path"])
            and label_rel != Path(original["label_path"]), "A10 file collides with its original image or label")
    require(row["published_sha256"] not in held_out_hashes and row["label_sha256"] == EMPTY_SHA256,
            "A10 asset collides with a held-out image or lacks exact empty-label evidence")
    require(sha256_file(image) == row["published_sha256"] and label.read_bytes() == b"",
            "A10 published pixel or empty-label bytes differ")
    return {"source_id": source, "group_id": row["group_id"], "split": "train",
            "image_path": image_rel.as_posix(), "label_path": label_rel.as_posix(),
            "image_sha256": row["published_sha256"], "label_sha256": row["label_sha256"], "instance_count": 0}


def reference_descriptor(split_rows, root):
    refs = {}
    for split in ("val", "test"):
        path = root / "images" / split
        refs[split] = {"image_path": path.as_posix(), "label_path": (root / "labels" / split).as_posix(),
                       "images": len(split_rows[split]), "labels": len(split_rows[split]),
                       "instances": sum(int(r["instance_count"]) for r in split_rows[split].values()),
                       "basis": "HASH_PINNED_SPLIT_MANIFEST", "images_decoded_this_run": 0,
                       "file_hashes_verified_this_run": 0}
    return refs


def config_for(arm, root):
    manifest_name, _ = ARM_CONFIGS[arm]
    return {"train": manifest_name, "val": (root / "images" / "val").as_posix(),
            "test": (root / "images" / "test").as_posix(), "nc": len(CLASSES), "names": CLASSES}


def write_configs(config_dir, baseline, derived, eval_root):
    for arm, (list_name, yaml_name) in ARM_CONFIGS.items():
        rows = baseline if arm == "e0" else baseline + derived
        paths = [f"./../dataset/{row['image_path']}" for row in rows]
        write_exclusive(config_dir / list_name, ("\n".join(paths) + "\n").encode("utf-8"))
        # JSON is a YAML mapping. Omit path so Ultralytics resolves train from the config's directory.
        write_json_exclusive(config_dir / yaml_name, config_for(arm, eval_root))


def verify_references(root, sealed):
    """Check exact VAL and TEST file members and hashes on the training host, without decoding."""
    require(sha256_file(root / "split_manifest.csv") == EXPECTED_MANIFEST_SHA256,
            "Training host frozen split manifest differs")
    require(load_class_mapping(root / "class_mapping.yaml") == dict(enumerate(CLASSES)),
            "Training host frozen class IDs differ")
    checked = {}
    for split in ("val", "test"):
        expected = {}
        for row in sealed[split].values():
            stem, suffix = row["stem"], row["image_suffix"]
            expected[f"images/{split}/{stem}{suffix}"] = row["sha256"]
            expected[f"labels/{split}/{stem}.txt"] = row["label_sha256"]
        observed = {p.relative_to(root).as_posix() for folder in ("images", "labels")
                    for p in (root / folder / split).iterdir() if p.is_file()}
        require(observed == expected.keys(), f"Frozen {split.upper()} file membership differs")
        require(all(sha256_file(root / name) == digest for name, digest in expected.items()),
                f"Frozen {split.upper()} pixel or label hash differs")
        checked[split] = len(expected)
    return checked


def verify_paths(config_dir, release_root, dataset_rows, eval_root):
    for arm, (list_name, yaml_name) in ARM_CONFIGS.items():
        config = json.loads((config_dir / yaml_name).read_text(encoding="utf-8"))
        require(config == config_for(arm, eval_root), "YOLO config class mapping or immutable split references differ")
        actual = (config_dir / list_name).read_text(encoding="utf-8").splitlines()
        rows = dataset_rows["baseline"] if arm == "e0" else dataset_rows["baseline"] + dataset_rows["derived"]
        require(actual == [f"./../dataset/{r['image_path']}" for r in rows], "TRAIN entry list differs")
        resolved = {(config_dir / rel[2:]).resolve(strict=True) for rel in actual}
        require(len(resolved) == len(actual)
                and resolved == {release_root / "dataset" / r["image_path"] for r in rows},
                "TRAIN entry repeats, crosses arms or escapes its released dataset")


def verify_inventory(inventory, sealed):
    """Check both arms against the sealed source splits before reading image bytes."""
    train = sealed["train"]
    baseline, derived = inventory["baseline"], inventory["derived"]
    require(len(baseline) == len(train) and {r["source_id"] for r in baseline} == train.keys(),
            "Original TRAIN inventory differs from the pinned source set")
    for row in baseline:
        frozen = train[row["source_id"]]
        compared = (row["group_id"], row["split"], row["image_sha256"], row["label_sha256"], row["instance_count"])
        expected = (frozen["group_id"], "train", frozen["sha256"], frozen["label_sha256"], int(frozen["instance_count"]))
        require(compared == expected, "Original TRAIN image or label identity differs")
    held_out = {row["sha256"] for split in ("val", "test") for row in sealed[split].values()}
    unmodified = {row["image_sha256"] for row in baseline}
    seen_sources, seen_hashes = set(), held_out | unmodified
    for row in derived:
        sid, digest = row["source_id"], row["image_sha256"]
        require(sid in train and sid not in seen_sources and digest not in seen_hashes,
                "Derived TRAIN inventory is not unique or intersects frozen original images")
        require(row["group_id"] == train[sid]["group_id"] and row["split"] == "train"
                and row["label_sha256"] == EMPTY_SHA256 and row["instance_count"] == 0,
                "Derived TRAIN identity or empty-label policy differs")
        seen_sources.add(sid)
        seen_hashes.add(digest)


def verify_local_parser(dataset, rows, wheel):
    require(not any(p.suffix.lower() in {".cache", ".npy"} for p in dataset.rglob("*") if p.is_file()),
            "Unexpected dataset cache or stale NumPy image")
    if wheel is not None:
        img2label, parser, numpy_version = load_ultralytics_parser(wheel)
        counts = verify_materialized(dataset, rows, img2label, parser)
        counts.setdefault("empty_labels", 0)
        return {"method": "PINNED_ULTRALYTICS_CPU_NUMPY_PIL", "version": ULTRALYTICS_VERSION,
                "wheel_sha256": WHEEL_SHA256, "numpy": numpy_version, "pillow": Image.__version__, **counts}
    counts = Counter(empty_labels=0, positive_labels=0, segmentation_instances=0)
    observed_paths = set()
    for row in rows:
        image, label = [under(dataset, row[key]) for key in ("image_path", "label_path")]
        require(image not in observed_paths and label not in observed_paths, "Dataset paths repeat")
        require(sha256_file(image) == row["image_sha256"] and sha256_file(label) == row["label_sha256"],
                "Published TRAIN image or label hash differs")
        rel = image.relative_to(dataset / "images")
        require(label == (dataset / "labels" / rel).with_suffix(".txt"), "Wrong Ultralytics label path")
        with Image.open(image) as decoded:
            decoded.load()
        polygons = label.read_text(encoding="utf-8").splitlines()
        require(len(polygons) == row["instance_count"], "TRAIN segmentation instance count differs")
        if not polygons:
            require(label.stat().st_size == 0, "Expected explicit zero-byte label")
        for line in polygons:
            fields = line.split()
            coords = [float(v) for v in fields[1:]]
            require(len(coords) >= 6 and len(coords) % 2 == 0 and fields[0] in {str(c) for c in range(len(CLASSES))}
                    and all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in coords), "Invalid segmentation format")
        counts["empty_labels" if not polygons else "positive_labels"] += 1
        counts["segmentation_instances"] += len(polygons)
        observed_paths.update((image, label))
    require(observed_paths == {p for p in dataset.rglob("*") if p.is_file()}, "TRAIN dataset has unindexed files")
    return {"method": "PIL_DECODE_AND_YOLO_SEGMENTATION_SYNTAX", **counts}


def arm_counts(rows):
    return {"train_images": len(rows), "train_labels": len(rows),
            "original_images": sum(r["instance_count"] > 0 for r in rows),
            "explicit_empty_labels": sum(r["instance_count"] == 0 for r in rows),
            "segmentation_instances": sum(r["instance_count"] for r in rows)}


def build_dataset_entries(asset_root, output, frozen_rows, wheel=None, *, eval_root=EVAL_ROOT):
    """Build the two training entry points and check data without model operations."""
    asset_root, output = Path(asset_root).resolve(strict=True), Path(output).resolve(strict=True)
    require(output.parent == asset_root, "A10 must use a dedicated directory directly under the private asset root")
    dataset, configs = output / "dataset", output / "configs"
    train_root, exported, _ = verify_train(asset_root)
    split_manifest = train_root / "frozen_split_manifest.csv"
    sealed = load_frozen_split(split_manifest)
    held_out_hashes = {r["sha256"] for split in ("val", "test") for r in sealed[split].values()}
    source_ids = {r["source_id"] for r in frozen_rows}
    require(len(source_ids) == len(frozen_rows) and source_ids <= sealed["train"].keys(),
            "A10 source repeats or is outside TRAIN")
    eval_root = PurePosixPath(Path(eval_root).as_posix())
    require(eval_root.is_absolute() or str(eval_root)[1:3] == ":/", "Frozen VAL/TEST path must be absolute")
    (dataset / "images" / "base").mkdir(parents=True)
    (dataset / "labels" / "base").mkdir(parents=True)
    baseline = baseline_rows(train_root, exported, dataset)
    records = {"baseline": list(baseline.values()),
               "derived": [bind_entry(output, dataset, row, baseline[row["source_id"]], held_out_hashes)
                           for row in sorted(frozen_rows, key=lambda r: r["source_id"])]}
    # Store only relocation-safe loader fields in the runtime inventory.
    for item in records["baseline"]:
        for key in ("source_image_path", "annotation_path", "annotation_sha256"):
            item.pop(key)
    verify_inventory(records, sealed)
    configs.mkdir(exist_ok=True)
    copy_verified(split_manifest, configs / "frozen_split_manifest.csv", EXPECTED_MANIFEST_SHA256)
    write_json_exclusive(configs / "train_entry_files.json", records)
    write_configs(configs, records["baseline"], records["derived"], eval_root)
    verify_paths(configs, output, records, eval_root)
    prior_wheel = asset_root / WHEEL_RELATIVE
    parser = verify_local_parser(dataset, records["baseline"] + records["derived"],
                                 Path(wheel) if wheel is not None else prior_wheel if prior_wheel.is_file() else None)
    require((parser["positive_labels"], parser["segmentation_instances"], parser["empty_labels"])
            == (EXPECTED_TRAIN_IMAGES, EXPECTED_TRAIN_INSTANCES, len(frozen_rows)), "Parsed TRAIN targets differ")
    evaluation = reference_descriptor(sealed, eval_root)
    result = {"status": "PASS_TRAIN_DATA_READ_ONLY_CHECK", "frozen_split_sha256": EXPECTED_MANIFEST_SHA256,
              "arms": {"e0": arm_counts(records["baseline"]), "e1": arm_counts(records["baseline"] + records["derived"])},
              "evaluation_references": evaluation, "train_group_count": len({r["group_id"] for r in records["baseline"]}),
              "class_names": CLASSES, "parser": parser, "optimizer_updates": 0,
              "torch_dataloader_executed": False, "held_out_inference_runs": 0}
    write_json_exclusive(output / "audits" / "dataloader_dry_run.json", result)
    return result


def installed_loader_check(output, inventory):
    """Exercise the installed segmentation DataLoader on CPU with isolated temporary caches."""
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import YOLODataset

    require(ultralytics.__version__ == ULTRALYTICS_VERSION, "Installed Ultralytics must match the pinned parser")

    class FreshCacheDataset(YOLODataset):
        def __init__(self, *args, checked_cache, **kwargs):
            self.checked_cache = checked_cache
            super().__init__(*args, **kwargs)

        def _load_or_scan_cache(self, cache_path, cache_hash):
            return super()._load_or_scan_cache(self.checked_cache, cache_hash)

    results = {}
    with TemporaryDirectory(prefix="exp14_a10_reader_") as scratch:
        for arm, (list_name, _) in ARM_CONFIGS.items():
            dataset = FreshCacheDataset(
                img_path=str(output / "configs" / list_name), imgsz=640, batch_size=8,
                augment=False, hyp=DEFAULT_CFG, rect=False, cache=False, task="segment",
                data={"names": dict(enumerate(CLASSES)), "channels": 3},
                checked_cache=Path(scratch) / f"{arm}.cache",
            )
            rows = inventory["baseline"] if arm == "e0" else inventory["baseline"] + inventory["derived"]
            expected = {output / "dataset" / r["image_path"]: r["instance_count"] for r in rows}
            actual = {Path(r["im_file"]).resolve(): len(r["segments"]) for r in dataset.labels}
            require(len(actual) == len(dataset) and actual == expected
                    and all(len(r["cls"]) == len(r["segments"]) for r in dataset.labels),
                    "Installed loader filtered or altered TRAIN instances")
            loader = build_dataloader(dataset, batch=8, workers=0, shuffle=False, device="cpu")
            batch = next(iter(loader))
            require(batch["img"].shape[1:] == (3, 640, 640) and batch["img"].device.type == "cpu",
                    "CPU read did not return standardized segmentation tensors")
            backgrounds = [i for i, row in enumerate(dataset.labels) if not len(row["segments"])]
            if backgrounds:
                require(len(dataset[backgrounds[0]]["cls"]) == 0, "Explicit empty TRAIN target did not yield zero classes")
            results[arm] = {"images": len(dataset), "empty_targets": len(backgrounds), "first_batch_size": len(batch["img"])}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify",))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wheel", type=Path, help="optional pinned wheel for CPU parser replay")
    parser.add_argument("--frozen-eval-root", type=Path, help="verify original VAL and TEST bytes on the target host")
    parser.add_argument("--installed-loader", action="store_true", help="exercise the pinned installed CPU DataLoader")
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    configs = output / "configs"
    sealed = load_frozen_split(configs / "frozen_split_manifest.csv")
    inventory = json.loads((configs / "train_entry_files.json").read_text(encoding="utf-8"))
    verify_inventory(inventory, sealed)
    e0_config = json.loads((configs / ARM_CONFIGS["e0"][1]).read_text(encoding="utf-8"))
    configured_eval = PurePosixPath(e0_config["val"]).parents[1]
    verify_paths(configs, output, inventory, configured_eval)
    report = {"parser": verify_local_parser(output / "dataset", inventory["baseline"] + inventory["derived"], args.wheel),
              "optimizer_updates": 0}
    if args.frozen_eval_root:
        supplied = args.frozen_eval_root.resolve(strict=True)
        require(supplied == Path(str(configured_eval)).resolve(), "Supplied VAL/TEST root differs from generated YAML")
        report["frozen_evaluation_file_hashes"] = verify_references(supplied, sealed)
    if args.installed_loader:
        report["cpu_dataloader"] = installed_loader_check(output, inventory)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
