"""Model-free, TRAIN-only native dataset construction for EXP18.

Importing this module imports neither Torch nor a trainer.  The dataset factory
uses the pinned native scanner and transforms, with an exclusive label-cache
writer.  Its source data and previous experiment directories are read-only.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch
import uuid


RUNTIME_CONFIG = "/root/exp17_r5_gate_20261124_v5/runtime_config.json"
RUNTIME_CONFIG_SHA256 = "1d90672b5ae86d2d2c6a64bf7326bdfefe35ae473b86c6ef579ee66f67b9b763"
MANIFEST_SHA256 = "fe6a36601c6c8d90cfddfc19ab58335552c18653b66f3535e84c97b4c222dc22"
NATIVE_IDENTITY_SHA256 = "289848daad87bbb8601251a88305d58dd48f41963f7d6adff0c4777603b14c4e"
BASELINE_LEDGER_SHA256 = "91fffd24dc5f1a904b42e97375e24ac4536813ad680bc9a2e1793f60fe4dfaf5"
R5_REFERENCE_SHA256 = "0e785fa763507463bf6c33ceddd647e747790450c47bd3d536196a2a15c0e864"
TRAIN_LIST_SHA256 = "d0d28459164df14290041ba3975ac779ca46725076c0a33e1ac6a19a91f1b963"
CLASSES = ("Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion")
SIX_TENSORS = ("img", "cls", "bboxes", "masks", "sem_masks", "batch_idx")
DATA_FIELDS = ("step", "epoch", "images", "raw_targets", "augmented_targets",
               "ordered_sources_sha256", "augmented_tensors_sha256")
FIXED_BATCHES = {
    1: "802114bde743ce7c58e997be279987148f33d91898d43b511ad5040939edd929",
    7: "07cff77e68290e0281cc6f8ac49a54dde0dbb25600c5c4932a0d5ba7b712ff76",
    11: "5395b0c3d0d7e0d22d3eb546f78e02fab29c9298aa886977328b140af6ff3f08",
    21: "c93789fb512af377a6d0793f3199126cd23e89ab1844fd08ecf30654e85aa90a",
}
AUGMENTATION = {"mosaic": 0., "close_mosaic": 0, "mixup": 0., "cutmix": 0.,
                "copy_paste": 0., "rect": False, "multi_scale": 0., "degrees": 0.,
                "translate": .1, "scale": .5, "shear": 0., "perspective": 0.,
                "flipud": 0., "fliplr": .5, "hsv_h": .015, "hsv_s": .7,
                "hsv_v": .4, "bgr": 0.}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def epoch_seed(seed, epoch, *, horizon=200):
    """Extend the original epoch domain, preserving the original seed bytes."""
    require(type(seed) is int and seed >= 0 and type(epoch) is int
            and type(horizon) is int and horizon in (100, 200) and 0 <= epoch < horizon,
            "Invalid EXP18 epoch seed inputs")
    payload = f"exp14-a11-full:v1:{seed}:{epoch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")


def load_runtime_config(path=RUNTIME_CONFIG, expected_sha256=RUNTIME_CONFIG_SHA256):
    """Read a pinned historical config but carry no historical run authorization."""
    require(sha256(path) == expected_sha256 == RUNTIME_CONFIG_SHA256,
            "Historical runtime config identity changed")
    old = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = ("release", "manifest", "native_identity", "runtime_assets", "baseline_ledger",
              "input_reference", "fixed_batches")
    result = {name: copy.deepcopy(old[name]) for name in fields}
    for key, expected in (("manifest", MANIFEST_SHA256), ("native_identity", NATIVE_IDENTITY_SHA256),
                          ("baseline_ledger", BASELINE_LEDGER_SHA256),
                          ("input_reference", R5_REFERENCE_SHA256)):
        require(sha256(result[key]) == expected, "Pinned metadata identity changed: " + key)
    result["manifest_sha256"] = MANIFEST_SHA256
    result["native_identity_sha256"] = NATIVE_IDENTITY_SHA256
    result["runtime_config_sha256"] = expected_sha256
    result["historical_runtime_config"] = str(path)
    result["train_list"] = str(Path(result["release"]) / "configs/train_e0_manifest.txt")
    require(sha256(result["train_list"]) == TRAIN_LIST_SHA256, "Frozen TRAIN list changed")
    require({int(s["batch_position"]): s["sha256"] for s in result["fixed_batches"]} == FIXED_BATCHES,
            "Fixed CPU packet whitelist changed")
    return result


def exclusive_label_cache(prefix, path, value, version):
    """Native cache callback without replacement or cleanup of failed writes."""
    import numpy as np
    target = Path(path)
    require(not target.is_symlink(), "Label cache target is a symlink")
    # The native scanner returns this same dictionary to get_labels(), which
    # removes its version field. Preserve that in-memory contract as well.
    value["version"] = version
    saved = copy.deepcopy(value)
    with target.open("xb") as stream:
        np.save(stream, saved)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass
class NumpyTensor:
    """An IPC envelope: avoid Torch tensor FD/resource_sharer transport."""
    array: Any


def _numpy_transport(value):
    import torch
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "Worker returned a non-CPU tensor")
        return NumpyTensor(value.detach().numpy())
    if isinstance(value, dict):
        return {k: _numpy_transport(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_numpy_transport(v) for v in value)
    return value


def restore_native_batch(value):
    import torch
    if isinstance(value, NumpyTensor):
        return torch.from_numpy(value.array)
    if isinstance(value, dict):
        return {k: restore_native_batch(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(restore_native_batch(v) for v in value)
    return value


def numpy_native_collate(samples):
    from ultralytics.data.dataset import YOLODataset
    return _numpy_transport(YOLODataset.collate_fn(samples))


def init_worker(worker_id):
    # Same worker seeding as the historical two-worker native loader.
    from ultralytics.data.build import seed_worker
    seed_worker(worker_id)


def new_epoch_loader(dataset, epoch, *, seed=42, horizon=200, numpy_transport=False):
    import torch
    from torch.utils.data import DataLoader
    require(os.name == "posix", "The pinned native replay requires Linux fork workers")
    generator = torch.Generator(device="cpu").manual_seed(epoch_seed(seed, epoch, horizon=horizon))
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2,
                      collate_fn=numpy_native_collate if numpy_transport else dataset.collate_fn,
                      pin_memory=False, drop_last=False, persistent_workers=False,
                      generator=generator, worker_init_fn=init_worker, multiprocessing_context="fork",
                      prefetch_factor=1)


def cpu_rng_snapshot():
    import random
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(), "CPU replay found initialized CUDA")
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].copy(), *state[2:]),
            "torch_cpu": torch.get_rng_state().clone()}


def cpu_rng_equal(left, right):
    import numpy as np
    import torch
    return (left["python"] == right["python"] and left["numpy"][0] == right["numpy"][0]
            and np.array_equal(left["numpy"][1], right["numpy"][1])
            and left["numpy"][2:] == right["numpy"][2:]
            and torch.equal(left["torch_cpu"], right["torch_cpu"]))


def native_batch_digest(batch):
    """Byte-identical to ja.batch_digest without importing its trainer module."""
    import torch
    h = hashlib.sha256()
    for key in SIX_TENSORS:
        tensor = batch[key]
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu",
                "Native six-tensor fingerprint requires CPU tensors")
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(key.encode())
        h.update(hashlib.sha256(raw).hexdigest().encode())
    return h.hexdigest()


def tensor_summary(value):
    raw = value.detach().contiguous().numpy()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "bytes_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}


def project_reference_rows(rows):
    """Keep exactly the historical fields actually compared by CPU replay.

    The complete input file is hashed before parsing. The large old ABN
    coverage dictionaries are not needed for native-byte/trace comparisons and
    must not be inherited by every new fork worker.
    """
    result = []
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("native"), dict)
                and isinstance(row.get("traces"), list), "Malformed native R5 reference")
        result.append({"native": row["native"], "traces": row["traces"]})
    return result


def release_cpu_memory():
    """Release unreachable objects / unused glibc arenas without RNG draws.

    This is resource housekeeping only: no file, native batch, parameter, or
    augmentation state is changed. Failed/partial experiment files remain.
    """
    import ctypes
    import gc

    def rss_kib():
        status = Path("/proc/self/status")
        if not status.exists():
            return None
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
        return None

    before = rss_kib()
    unreachable = gc.collect()
    supported, released = False, None
    if os.name == "posix":
        try:
            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            released = int(trim(0))
            supported = True
        except (AttributeError, OSError):
            pass
    return {"gc_unreachable": unreachable, "malloc_trim_supported": supported,
            "malloc_trim_return": released, "rss_before_KiB": before, "rss_after_KiB": rss_kib()}


def prepare_fork_memory():
    """Share imported CPU helpers and protect the old heap from fork COW.

    Permanent-generation objects are not rescanned by later collections in
    either parent or workers. New batch cycles remain normally collectable.
    This process intentionally keeps that permanent generation until exit.
    """
    import gc
    import importlib
    names = ("scipy.optimize", "tools.exp18.pairing", "tools.exp18.geometry",
             "tools.exp15.regions", "tools.exp17.engineering.augmentation",
             "torch.multiprocessing.reductions", "multiprocessing.resource_sharer",
             "multiprocessing.connection")
    before = cpu_rng_snapshot()
    loaded = []
    for name in names:
        module = importlib.import_module(name)
        loaded.append({"module": name, "version": getattr(module, "__version__", None)})
    require(cpu_rng_equal(before, cpu_rng_snapshot()), "CPU helper preload consumed native RNG")
    first = release_cpu_memory()
    require(hasattr(gc, "freeze") and hasattr(gc, "get_freeze_count"),
            "Fork memory policy requires CPython permanent-generation support")
    previous_count = gc.get_freeze_count()
    gc.freeze()
    frozen_count = gc.get_freeze_count()
    second = release_cpu_memory()
    require(cpu_rng_equal(before, cpu_rng_snapshot()), "CPU heap freeze consumed native RNG")
    return {"preloaded_modules": loaded, "gc_freeze_count_before": previous_count,
            "gc_freeze_count": frozen_count, "permanent_generation_retained_until_exit": True,
            "before_freeze_release": first, "after_freeze_release": second,
            "new_batch_cycles_still_collected": True}


def load_references(config):
    require(sha256(config["baseline_ledger"]) == BASELINE_LEDGER_SHA256
            and sha256(config["input_reference"]) == R5_REFERENCE_SHA256,
            "Historical input reference changed")
    with Path(config["baseline_ledger"]).open(encoding="utf-8-sig", newline="") as stream:
        j1 = list(csv.DictReader(stream))
    r5 = project_reference_rows(json.loads(Path(config["input_reference"]).read_text()))
    require(len(j1) == len(r5) == 2100, "Historical data reference is incomplete")
    for i, row in enumerate(j1, 1):
        for key in ("step", "epoch", "images", "raw_targets", "augmented_targets"):
            row[key] = int(row[key])
        require(row["step"] == i and all(row[k] == r5[i - 1]["native"][k] for k in DATA_FIELDS),
                "J1 and R5 native references disagree")
    return j1, r5


@dataclass
class DatasetContext:
    dataset: Any
    index: Any
    groups: dict
    expected: dict
    dataset_root: Path
    source_files: tuple
    cache_path: Path
    config: dict

    def postcheck(self):
        for path, expected_sha, size, modified in self.source_files:
            stat = path.stat()
            require(stat.st_size == size and stat.st_mtime_ns == modified and sha256(path) == expected_sha,
                    "Read-only source changed: " + str(path))


@dataclass(frozen=True)
class NativeROIView:
    """Original ROI tensors plus immutable evidence, without A image pixels."""
    batch: dict
    regions: Any
    image_shape: tuple
    a_image_summaries: tuple
    six_tensors: dict
    retained_tensors: dict
    trace_sha256: tuple
    evidence_sha256: str


def _view_evidence(view):
    return {"image_shape": list(view.image_shape), "A_images": list(view.a_image_summaries),
            "six_tensors": view.six_tensors, "trace_sha256": list(view.trace_sha256)}


def retain_native_roi_view(batch, groups):
    """Observe the real native batch before releasing image/semantic pixels."""
    from tools.exp15.regions import inspect_regions
    require("img" in batch and "sem_masks" in batch, "ROI view requires the original native six tensors")
    regions = inspect_regions(batch, groups)
    shape = tuple(int(v) for v in batch["img"].shape[-2:])
    images = tuple(tensor_summary(image) for image in batch["img"])
    six = {name: tensor_summary(batch[name]) for name in SIX_TENSORS}
    keys = ("masks", "cls", "bboxes", "batch_idx", "im_file", "j2_replay")
    retained = {name: batch[name] for name in keys}
    summaries = {name: six[name] for name in ("masks", "cls", "bboxes", "batch_idx")}
    traces = tuple(trace.sha256 for trace in batch["j2_replay"])
    evidence = {"image_shape": list(shape), "A_images": list(images),
                "six_tensors": six, "trace_sha256": list(traces)}
    return NativeROIView(retained, regions, shape, images, six, summaries, traces, digest(evidence))


def validate_precomputed_regions(batch, regions, groups, image_shape):
    """Recheck every original ROI ID/class/cell using retained masks, no fake img."""
    import numpy as np
    height, width = image_shape
    masks = batch["masks"].numpy()
    classes = batch["cls"].reshape(-1).numpy()
    slots = batch["batch_idx"].numpy()
    boxes = batch["bboxes"].numpy()
    require(height % 8 == width % 8 == 0 and masks.shape == (len(batch["im_file"]), height // 4, width // 4),
            "Retained native mask/image shape binding changed")
    require(slots.shape == classes.shape and boxes.shape == (len(classes), 4)
            and np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            and np.isin(slots, np.arange(len(masks))).all() and np.isin(classes, np.arange(7)).all(),
            "Retained native ROI coordinates/classes/slots changed")
    expected_groups = np.asarray([groups[str(Path(name).resolve(strict=True))] for name in batch["im_file"]])[slots.astype(np.int64)]
    require(np.array_equal(regions.slots, slots.astype(np.int64))
            and np.array_equal(regions.classes, classes.astype(np.int64))
            and np.array_equal(regions.groups, expected_groups), "Precomputed ROI identity changed")
    cells = np.zeros(len(classes), dtype=np.int64)
    ordinals = np.empty(len(classes), dtype=np.int64)
    for slot, mask in enumerate(masks):
        indexes = np.flatnonzero(slots == slot)
        ordinals[indexes] = np.arange(len(indexes))
        require(np.isin(mask, np.arange(len(indexes) + 1)).all(), "Retained native mask contains an unbound ROI ID")
        for local, row in enumerate(indexes, 1):
            y, x = np.nonzero(mask == local)
            cells[row] = len(np.unique((y // 2) * (mask.shape[1] // 2) + x // 2))
    require(np.array_equal(regions.ordinals, ordinals) and np.array_equal(regions.cells, cells),
            "Precomputed native ROI ordinal/P3 cell count changed")
    return regions


def check_retained_roi_view(view, groups):
    require("img" not in view.batch and "sem_masks" not in view.batch,
            "CPU metadata view must not retain image or semantic-mask pixels")
    require(digest(_view_evidence(view)) == view.evidence_sha256,
            "Pre-drop original native image evidence was altered")
    require(all(tensor_summary(view.batch[name]) == summary for name, summary in view.retained_tensors.items()),
            "Retained native ROI tensor bytes were altered")
    require(tuple(trace.sha256 for trace in view.batch["j2_replay"]) == view.trace_sha256,
            "Retained native augmentation trace changed")
    return validate_precomputed_regions(view.batch, view.regions, groups, view.image_shape)


def build_dataset(config, output, *, horizon=200):
    """Construct the original native dataset, never a trainer or model."""
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.cfg import get_cfg
    import ultralytics.data.dataset as dataset_module
    from tools.exp17.engineering.data import TrustedPairIndex
    from tools.exp17.engineering.augmentation import IndexIdentityResolver, install_native_replay

    require(torch.__version__ == "2.8.0+cu128" and ultralytics.__version__ == "8.4.117",
            "Pinned native data runtime differs")
    require(not torch.cuda.is_initialized(), "Dataset construction must not initialize CUDA")
    release = Path(config["release"]).resolve(strict=True)
    dataset_root = release / "dataset"
    output = Path(output).resolve()
    require(not output.is_relative_to(release), "Dataset output intersects frozen release")
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / ("labels_" + uuid.uuid4().hex + ".cache")
    index = TrustedPairIndex.from_files(config["manifest"], config["native_identity"],
        expected_manifest_sha256=MANIFEST_SHA256, runtime_assets=config["runtime_assets"],
        native_dataset_root=dataset_root)
    require(len(index.train) == 668 and sum(len(s["native_labels"]) for s in index.train.values()) == 1266,
            "Native TRAIN population changed")
    expected, groups, checked = {}, {}, {}
    for source in index.train.values():
        require(source["split"] == "train", "Held-out source entered dataset")
        image = (dataset_root / source["native_image_relative"]).resolve(strict=True)
        label = (dataset_root / source["native_label_relative"]).resolve(strict=True)
        for path, wanted in ((image, source["a_sha256"]), (label, source["label_sha256"])):
            stat = path.stat()
            require(sha256(path) == wanted, "Frozen TRAIN byte identity changed: " + str(path))
            checked[str(path)] = (path, wanted, stat.st_size, stat.st_mtime_ns)
        with image.open("rb") as stream:
            if stream.read(2) == b"\xff\xd8":
                stream.seek(-2, os.SEEK_END)
                require(stream.read(2) == b"\xff\xd9", "Native scanner would rewrite incomplete JPEG")
        rows = source["native_labels"]
        wanted = "".join(r["native_label_line"] + "\n" for r in rows).encode()
        require(label.read_bytes() == wanted, "Native label ordinal identity changed")
        classes = tuple(r["class_id"] for r in rows)
        polygons = tuple(np.asarray(r["normalized_native_coordinates"], dtype=np.float32).reshape(-1, 2) for r in rows)
        expected[image] = (classes, polygons)
        groups[str(image)] = source["group_id"]

    overrides = {"task": "segment", "imgsz": 640, "batch": 32, "nbs": 32, "epochs": horizon,
                 "seed": 42, "deterministic": True, "cache": False, "single_cls": False,
                 "workers": 2, "fraction": 1., "mask_ratio": 4, "overlap_mask": True,
                 **AUGMENTATION}
    hyp = get_cfg(overrides=overrides)
    data = {"train": config["train_list"], "val": None, "test": None,
            "nc": 7, "channels": 3, "names": dict(enumerate(CLASSES))}

    class IsolatedDataset(dataset_module.YOLODataset):
        def _load_or_scan_cache(self, ignored_path, cache_hash):
            require(not cache_path.exists(), "Never replace a label cache")
            cache, reused = super()._load_or_scan_cache(cache_path, cache_hash)
            self.scan_counts = tuple(cache["results"])
            require(not reused and not cache["msgs"], "Native scan found stale cache or source warnings")
            return cache, reused

    with patch.object(dataset_module, "save_dataset_cache_file", exclusive_label_cache):
        dataset = IsolatedDataset(img_path=config["train_list"], imgsz=640, batch_size=32,
            augment=True, hyp=hyp, rect=False, cache=False, single_cls=False, stride=32,
            pad=0., prefix="EXP18 TRAIN: ", task="segment", classes=None, data=data, fraction=1.)
    seen = set()
    for label in dataset.labels:
        path = Path(label["im_file"]).resolve(strict=True)
        require(path in expected and path not in seen, "Native scan changed TRAIN membership")
        classes, polygons = expected[path]
        require(tuple(label["cls"].reshape(-1)) == classes and len(label["segments"]) == len(polygons)
                and all(np.array_equal(a, b) for a, b in zip(label["segments"], polygons)),
                "Native scan changed class or Polygon identity")
        seen.add(path)
    require(seen == expected.keys() and len(dataset) == 668
            and dataset.scan_counts == (668, 0, 0, 0, 668), "Native TRAIN scan count changed")
    install_native_replay(dataset, IndexIdentityResolver(index))
    # The no-mixing TRAIN stream visits each source exactly once per epoch.
    # Only inactive image-buffer retention changes, never sample/transform RNG.
    # Keep memory bounded under the actual 2-GiB AutoDL cgroup, not host RAM.
    dataset.max_buffer_length = 8
    context = DatasetContext(dataset, index, groups, expected, dataset_root,
                             tuple(checked.values()), cache_path, copy.deepcopy(config))
    context.postcheck()
    require(not torch.cuda.is_initialized(), "Native dataset initialized CUDA")
    return context


def native_evidence(context, batch, *, epoch, batch_index):
    full = [Path(name).resolve(strict=True) for name in batch["im_file"]]
    require(len(full) == len(set(full)) and all(p in context.expected for p in full),
            "Native batch changed TRAIN membership")
    paths = [p.relative_to(context.dataset_root).as_posix() for p in full]
    ordered = hashlib.sha256(json.dumps(paths, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return {"step": epoch * 21 + batch_index + 1, "epoch": epoch + 1, "images": len(full),
            "raw_targets": sum(len(context.expected[p][0]) for p in full),
            "augmented_targets": int(batch["cls"].numel()), "ordered_sources_sha256": ordered,
            "augmented_tensors_sha256": native_batch_digest(batch)}


def compare_historical(actual, traces, references, r5_references):
    step = actual["step"]
    if step > 2100:
        return False
    require(all(actual[k] == references[step - 1][k] == r5_references[step - 1]["native"][k]
                for k in DATA_FIELDS), "Native data differs from frozen J1/R5 at step " + str(step))
    require([t.sha256 for t in traces] == r5_references[step - 1]["traces"],
            "Native augmentation trace differs from R5 at step " + str(step))
    return True


def load_fixed_cpu_packet(spec):
    """The only permitted real .pt loads in model-free EXP18 construction."""
    import torch
    position = int(spec["batch_position"])
    require(position in FIXED_BATCHES and spec["sha256"] == FIXED_BATCHES[position]
            and sha256(spec["path"]) == FIXED_BATCHES[position], "CPU packet is outside the fixed whitelist")
    from .guards import active_guard
    guard = active_guard()
    if guard is not None:
        guard.allow_cpu_load(spec["path"], spec["sha256"], kind="fixed_TRAIN_CPU_batch")
    packet = torch.load(spec["path"], map_location="cpu", weights_only=False)
    require(packet["schema"] == "EXP17_J2_NATIVE_CPU_BATCH_CACHE_V1" and packet["epoch"] == 0
            and packet["batch_position"] == position and packet["step"] == position,
            "Fixed CPU packet schema/position changed")
    require(native_batch_digest(packet["batch"]) == spec["native_six_tensor_sha256"],
            "Fixed CPU native bytes changed")
    return packet
