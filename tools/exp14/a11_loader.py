"""Read every A10 TRAIN sample through Ultralytics without training or model inference."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from tools.exp13_cloud.g3.g3_batch_executor import write_json_exclusive
from tools.exp14.a10_dataset_entries import (
    ARM_CONFIGS, EVAL_ROOT, ULTRALYTICS_VERSION, load_frozen_split, verify_inventory, verify_paths,
)
from tools.exp14.audit_admission import require
from tools.exp14.label_plan import CLASSES

BATCH = 8
IMAGE_SIZE = 640


def read_expected_labels(dataset_root, rows):
    expected = {}
    for row in rows:
        image = (dataset_root / row["image_path"]).resolve(strict=True)
        label = (dataset_root / row["label_path"]).resolve(strict=True)
        require(image not in expected, "TRAIN image path is repeated")
        with image.open("rb") as pixels:
            if pixels.read(2) == b"\xff\xd8":
                pixels.seek(-2, os.SEEK_END)
                require(pixels.read(2) == b"\xff\xd9", "Native JPEG verifier would rewrite an incomplete image")
        payload = label.read_bytes()
        records = [line.split() for line in payload.decode("utf-8").splitlines()]
        require(len(records) == row["instance_count"] and (records or payload == b""),
                "Native loader input lacks an exact positive or explicit empty label")
        classes = tuple(int(record[0]) for record in records)
        require(all(0 <= c < len(CLASSES) for c in classes), "Expected class ID is outside the frozen mapping")
        polygons = tuple(np.asarray(record[1:], dtype=np.float32).reshape(-1, 2) for record in records)
        require(all(len(p) >= 3 and np.isfinite(p).all() for p in polygons), "Expected label is not a finite polygon")
        expected[image] = (classes, polygons)
    return expected


def validate_dataset(dataset, expected):
    by_file = {}
    for label in dataset.labels:
        path = Path(label["im_file"]).resolve(strict=True)
        classes, polygons = expected[path]
        require(path not in by_file and tuple(label["cls"].reshape(-1)) == classes,
                "Native label scan repeated an image or changed a class")
        require(len(label["segments"]) == len(polygons) and
                all(np.array_equal(saved, read) for saved, read in zip(polygons, label["segments"])),
                "Native label scan dropped or changed a Polygon")
        by_file[path] = classes
    require(len(by_file) == len(dataset) == len(expected) and by_file.keys() == expected.keys(),
            "Native label scan dropped a TRAIN image")
    return by_file


def batch_counts(paths, expected, classes, indexes, boxes, masks, sem_masks, n, mask_size):
    """Check all collated targets against the source file occupying each batch slot."""
    require(len(paths) == len(set(paths)) == n and all(path in expected for path in paths),
            "Native collate repeated or changed batch image paths")
    require(classes.ndim == 2 and classes.shape[1] == 1 and indexes.shape == (len(classes),)
            and boxes.shape == (len(classes), 4) and np.isfinite(boxes).all()
            and ((0 <= indexes) & (indexes < n) & (indexes == np.floor(indexes))).all(),
            "Native collate has mismatched or invalid target slots")
    require(masks.shape == sem_masks.shape == (n, mask_size, mask_size)
            and np.isfinite(masks).all() and np.isfinite(sem_masks).all(), "Native mask batch shape or values differ")
    counts = [len(expected[path]) for path in paths]
    for slot, (path, count) in enumerate(zip(paths, counts)):
        # Overlap-mask formatting reorders instances by raster area within each image.
        require(Counter(classes[indexes == slot].reshape(-1)) == Counter(expected[path]),
                "Native collate dropped a target, changed its class or assigned it to another image")
        require(np.min(masks[slot]) >= 0 and np.max(masks[slot]) <= count
                and np.array_equal(masks[slot], np.floor(masks[slot]))
                and (count or (not np.any(masks[slot]) and not np.any(sem_masks[slot]))),
                "Native collate returned invalid or nonempty background masks")
    background_count = counts.count(0)
    kind = "all_background" if background_count == n else "original_repair_mixed" if background_count else "all_original"
    return kind, sum(counts), background_count


def run_arm(dataset, loader, expected, torch, cuda, mask_size):
    images, kinds, examples = set(), Counter(), {}
    instances, backgrounds, tail = 0, 0, 0
    for number, batch in enumerate(loader, start=1):
        img = batch["img"]
        n = len(img)
        paths = [Path(name).resolve(strict=True) for name in batch["im_file"]]
        require(img.dtype == torch.uint8 and img.device.type == "cpu" and img.shape == (n, 3, IMAGE_SIZE, IMAGE_SIZE)
                and not (images & set(paths)), "Native iterator produced invalid or repeated TRAIN images")
        arrays = [batch[k].numpy() for k in ("cls", "batch_idx", "bboxes", "masks", "sem_masks")]
        kind, count, n_bg = batch_counts(paths, expected, *arrays, n, mask_size)
        if kind not in examples:
            examples[kind] = {"batch_number": number, "images": n, "original_images": n - n_bg,
                              "background_images": n_bg, "instances": count,
                              "image_tensor_shape": list(img.shape), "mask_tensor_shape": list(batch["masks"].shape),
                              "semantic_mask_tensor_shape": list(batch["sem_masks"].shape)}
            if cuda:
                keys = ("img", "cls", "bboxes", "masks", "sem_masks", "batch_idx")
                transferred = {key: batch[key].to("cuda:0") for key in keys}
                torch.cuda.synchronize()
                require(all(t.device.type == "cuda" and torch.equal(t.cpu(), batch[k]) for k, t in transferred.items()),
                        "CUDA TRAIN tensor round trip altered data")
                examples[kind]["cuda_tensor_roundtrip"] = True
        images.update(paths)
        kinds[kind] += 1
        instances += count
        backgrounds += n_bg
        tail = n
    require(images == expected.keys() and len(dataset) == len(images)
            and instances == sum(map(len, expected.values())) and backgrounds == sum(not v for v in expected.values())
            and sum(kinds.values()) == len(loader) and tail == (len(images) % BATCH or BATCH),
            "Native iterator did not visit all frozen TRAIN images and targets including the last batch")
    return {"images": len(images), "instances": instances, "explicit_empty_labels": backgrounds,
            "batches": sum(kinds.values()), "last_batch_images": tail, "batch_types": dict(kinds), "examples": examples}


def installed_precheck(release, configs, eval_root, cuda):
    sealed = load_frozen_split(release / "configs" / "frozen_split_manifest.csv")
    inventory = json.loads((release / "configs" / "train_entry_files.json").read_text(encoding="utf-8"))
    verify_inventory(inventory, sealed)
    verify_paths(configs, release, inventory, PurePosixPath(eval_root.as_posix()))
    data_root = release / "dataset"
    require(not any(p.suffix.lower() in {".cache", ".npy"} for p in data_root.rglob("*") if p.is_file()),
            "Unexpected loader cache exists under frozen TRAIN")
    expected = {arm: read_expected_labels(data_root, inventory["baseline"] + (inventory["derived"] if arm == "e1" else []))
                for arm in ARM_CONFIGS}

    with TemporaryDirectory(prefix="exp14_a11_precheck_") as temp, patch.dict(os.environ, {
        "YOLO_CONFIG_DIR": temp, "MPLCONFIGDIR": temp, "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false",
    }):
        import torch
        import ultralytics
        from ultralytics.cfg import DEFAULT_CFG
        from ultralytics.data.build import build_dataloader
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.utils import check_det_dataset

        require(ultralytics.__version__ == ULTRALYTICS_VERSION, "Installed Ultralytics differs from the frozen version")
        require(not cuda or torch.cuda.is_available(), "CUDA requested but unavailable")
        require(DEFAULT_CFG.overlap_mask and IMAGE_SIZE % DEFAULT_CFG.mask_ratio == 0,
                "Native mask rendering defaults differ from the precheck")

        class IsolatedCacheDataset(YOLODataset):
            def __init__(self, checked_cache, **kwargs):
                self.checked_cache = checked_cache
                super().__init__(**kwargs)

            def _load_or_scan_cache(self, cache_path, cache_hash):
                cache, reused = super()._load_or_scan_cache(self.checked_cache, cache_hash)
                self.scan_counts = tuple(cache["results"])
                require(not reused and not cache["msgs"], "Native scan used a cache or detected image/label changes")
                return cache, reused

        dataset_results = {}
        for arm, (list_name, config_name) in ARM_CONFIGS.items():
            # This path check requires no plot fonts. Skip the framework's optional font download.
            with patch("ultralytics.data.utils.check_font", return_value=None):
                data = check_det_dataset(str(configs / config_name), autodownload=False)
            paths = (Path(data["train"]), Path(data["val"]), Path(data["test"]))
            correct = ((configs / list_name).resolve(), (eval_root / "images" / "val").resolve(),
                       (eval_root / "images" / "test").resolve())
            require(paths == correct and data["names"] == dict(enumerate(CLASSES)) and data["nc"] == len(CLASSES),
                    "Native YAML resolver changed a dataset path or class mapping")
            dataset = IsolatedCacheDataset(
                checked_cache=Path(temp) / f"{arm}.cache", img_path=str(paths[0]), imgsz=IMAGE_SIZE,
                batch_size=BATCH, augment=False, hyp=DEFAULT_CFG, rect=False, cache=False, task="segment",
                data=data, prefix=f"A11 {arm.upper()}: ",
            )
            observed = validate_dataset(dataset, expected[arm])
            marks = (len(observed), 0, sum(not v for v in observed.values()), 0, len(observed))
            require(dataset.scan_counts == marks, "Native scan did not distinguish explicit empty and missing labels")
            loader = build_dataloader(dataset, batch=BATCH, workers=0, shuffle=False, device="cpu")
            dataset_results[arm] = run_arm(dataset, loader, observed, torch, cuda, IMAGE_SIZE // DEFAULT_CFG.mask_ratio)

        require(set(dataset_results["e0"]["batch_types"]) == {"all_original"} and
                set(dataset_results["e1"]["batch_types"]) == {"all_original", "original_repair_mixed", "all_background"},
                "Required native batch types were not represented")
        require(not any(p.suffix.lower() in {".cache", ".npy"} for p in data_root.rglob("*") if p.is_file()),
                "Native reader modified the frozen TRAIN cache state")
        return {"status": "PASS_NATIVE_READ_ONLY_ULTRALYTICS_DATALOADER", "ultralytics": ultralytics.__version__,
                "torch": torch.__version__, "batch_size": BATCH, "image_size": IMAGE_SIZE, "classes": CLASSES,
                "isolated_caches": True, "cuda_tensor_roundtrip": cuda, "arms": dataset_results,
                "training_performed": False, "backward_called": False, "optimizer_updates": 0,
                "val_model_inference_runs": 0, "test_model_inference_runs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--configs", type=Path, help="optional runtime configuration directory")
    parser.add_argument("--frozen-eval-root", type=Path, default=Path(EVAL_ROOT))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    configs = args.configs.resolve(strict=True) if args.configs else release / "configs"
    eval_root = args.frozen_eval_root.absolute()
    require(eval_root.is_dir(), "Frozen evaluation root is unavailable")
    require(not args.report.resolve().is_relative_to(release), "Write the DataLoader report outside the immutable release")
    data = installed_precheck(release, configs, eval_root, args.cuda)
    write_json_exclusive(args.report, data)
    print(json.dumps(data, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
