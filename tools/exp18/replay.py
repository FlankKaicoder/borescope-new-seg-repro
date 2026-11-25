"""Full-horizon CPU data replay, closed matching, and immutable epoch ledgers.

No trainer, real model, checkpoint weights, VAL image or TEST file is needed.
Only the currently processed native A batch and its sparse N images are live.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import time
import uuid

from .data import (DATA_FIELDS, SIX_TENSORS, build_dataset, check_retained_roi_view,
                   compare_historical, cpu_rng_equal,
                   cpu_rng_snapshot, digest, load_references, native_evidence,
                   new_epoch_loader, prepare_fork_memory, release_cpu_memory, require,
                   restore_native_batch, retain_native_roi_view, sha256, tensor_summary,
                   validate_precomputed_regions)


class DecodedAssets:
    """Bounded read-only RAM cache, with verified hashes before first decode."""
    def __init__(self, max_bytes=32 * 1024 ** 2):
        self.max_bytes = max_bytes
        self.items = OrderedDict()
        self.bytes = 0
        self.decodes = Counter()
        self.verified = {}

    def read(self, path, expected, kind):
        import cv2
        import numpy as np
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        key = (str(path), expected, kind)
        if key in self.items:
            old, value = self.items.pop(key)
            require(old == stamp, "Frozen cached asset changed")
            self.items[key] = (old, value)
            return value
        require(sha256(path) == expected, "Frozen replay asset hash changed")
        flags = cv2.IMREAD_COLOR if kind == "N" else cv2.IMREAD_GRAYSCALE
        value = cv2.imread(str(path), flags)
        require(value is not None and value.dtype == np.uint8, "Frozen asset decode failed")
        if kind != "N":
            require(np.isin(value, [0, 1, 255]).all(), "Invalid frozen binary Valid mask")
        self.decodes[kind] += 1
        self.verified[str(path)] = (expected, stamp)
        if value.nbytes <= self.max_bytes:
            while self.items and self.bytes + value.nbytes > self.max_bytes:
                _, (_, old) = self.items.popitem(last=False)
                self.bytes -= old.nbytes
            self.items[key] = (stamp, value)
            self.bytes += value.nbytes
        return value

    def postcheck(self):
        for name, (expected, stamp) in self.verified.items():
            path = Path(name)
            stat = path.stat()
            require((stat.st_size, stat.st_mtime_ns) == stamp and sha256(path) == expected,
                    "Frozen N/Valid asset changed during replay")


@dataclass
class PreparedExposure:
    batch: dict
    a_regions: object
    normal_images: object
    normal_masks: object
    normal_regions: object
    plan: object
    collection: object
    coverage: dict
    native_rows: tuple
    replay_records: tuple


def _native_rows(batch, context, regions):
    import numpy as np
    rows, originals = [], []
    traces = batch.get("j2_replay")
    require(traces is not None and len(traces) == len(batch["im_file"]), "Missing native ReplayTrace")
    for original in np.flatnonzero(regions.cells > 0):
        slot, ordinal = int(regions.slots[original]), int(regions.ordinals[original])
        trace = traces[slot]
        source = context.index.source_for_image(batch["im_file"][slot])
        native = trace.native_target_rows[ordinal]
        require(native["source_id"] == source and native["native_ordinal"] == ordinal
                and native["mask_id"] == ordinal + 1
                and int(native["class_id"]) == int(regions.classes[original])
                and native["group_id"] == str(regions.groups[original]), "Native region identity changed")
        row = {"row_index": len(rows), "slot": slot, "source_id": source,
               "group_id": native["group_id"], "polygon_id": native["polygon_id"],
               "class_id": int(native["class_id"]), "annotation_ordinal": int(native["annotation_ordinal"]),
               "native_label_ordinal": int(native["native_label_ordinal"]),
               "native_post_format_ordinal": ordinal, "p3_cells": int(regions.cells[original]),
               "augmentation_sha256": trace.sha256}
        context.index.validate_native_region(row)
        rows.append(row)
        originals.append(int(original))
    return rows, originals


def prepare_exposure(batch, context, matching_config, *, epoch, batch_index,
                     decode_n=True, asset_cache=None, native_geometry=None, materialize_n=True,
                     precomputed_a_regions=None, image_shape=None, a_image_summaries=None):
    """Prepare every admitted region independently of historical B availability."""
    import numpy as np
    import torch
    from tools.exp15.regions import Regions, inspect_regions
    from tools.exp17.engineering.augmentation import array_sha, replay_normal, support_from_trace
    from .geometry import native_polygon_area_fraction, region_geometry
    from .pairing import MatchingConfig, build_matching, collect_candidates

    require(not torch.cuda.is_initialized(), "CPU preparation found CUDA initialized")
    before = cpu_rng_snapshot()
    if precomputed_a_regions is None:
        require("img" in batch, "Original native image tensor is required without precomputed ROI evidence")
        regions = inspect_regions(batch, context.groups)
    else:
        require(image_shape is not None, "Precomputed ROI preparation requires original image shape evidence")
        regions = validate_precomputed_regions(batch, precomputed_a_regions, context.groups, image_shape)
    if a_image_summaries is None:
        require("img" in batch, "Missing pre-drop native A image summaries")
        a_image_summaries = tuple(tensor_summary(image) for image in batch["img"])
    require(len(a_image_summaries) == len(batch["im_file"]), "Native A slot/image-summary count differs")
    rows, originals = _native_rows(batch, context, regions)
    assets = asset_cache if asset_cache is not None else DecodedAssets()
    if native_geometry is None:
        native_geometry = {label["polygon_id"]: {"source_area_fraction": native_polygon_area_fraction(
            label["normalized_native_coordinates"])} for source in context.index.train.values()
            for label in source["native_labels"]}
    support_by_row, slots = {}, {}
    geometry_records = []
    for row in rows:
        trusted = context.index.by_polygon.get(row["polygon_id"])
        if trusted is None:
            continue
        slot = row["slot"]
        trace = batch["j2_replay"][slot]
        if slot not in slots:
            source_valid = assets.read(context.index.asset_path(trusted, "source_valid"),
                                       trusted["source_valid_mask_sha256"], "source_valid")
            n_valid = assets.read(context.index.asset_path(trusted, "n_valid"),
                                  trusted["n_valid_mask_sha256"], "n_valid")
            require(source_valid.shape == n_valid.shape == tuple(trace.original_hw),
                    "A/N Valid masks differ from their frozen original frame")
            combined = np.ascontiguousarray(((source_valid > 0) & (n_valid > 0)).astype(np.uint8) * 255)
            support = support_from_trace(combined, trace)
            slots[slot] = {"trusted": trusted, "combined": combined, "support": support}
        saved = slots[slot]
        require(saved["trusted"]["n_sha256"] == trusted["n_sha256"], "One source has inconsistent N versions")
        ordinal = row["native_post_format_ordinal"]
        observed = saved["support"]["support"][ordinal]
        require(observed["polygon_id"] == row["polygon_id"], "Support attached to another Polygon")
        mask = batch["masks"][slot].numpy()
        g = region_geometry(mask, ordinal + 1, valid_full=saved["support"]["valid_full"])
        require(g.cells == row["p3_cells"] and g.support_complete == observed["support_complete"],
                "Independent P3 geometry and native Valid support disagree")
        support_by_row[row["row_index"]] = {"polygon_id": row["polygon_id"],
            "support_complete": bool(observed["support_complete"]),
            "mask_sha256": array_sha(np.ascontiguousarray(mask == ordinal + 1)), "p3_cells": g.cells}
        geometry_records.append({"a_row": row["row_index"], "polygon_id": row["polygon_id"],
                                 **g.to_dict()})
    collection = collect_candidates(rows, context.index.by_polygon, support_by_row, native_geometry)
    if isinstance(matching_config, dict):
        matching_config = MatchingConfig.from_protocol({"matching": matching_config})
    plan = build_matching(collection.candidates, matching_config, epoch=epoch, batch_index=batch_index)
    active_slots = list(plan.selected_a_slots)
    compact = {slot: n for n, slot in enumerate(active_slots)}
    selected = np.array([i for i, slot in enumerate(regions.slots) if int(slot) in compact], dtype=np.int64)
    normal_regions = Regions(np.array([compact[int(regions.slots[i])] for i in selected], dtype=np.int64),
        regions.ordinals[selected].copy(), regions.classes[selected].copy(), regions.groups[selected].copy(),
        regions.cells[selected].copy(), regions.foreground_fraction[selected].copy(),
        max((int(np.count_nonzero(regions.slots == slot)) for slot in active_slots), default=0))
    original_to_n = {int(original): n for n, original in enumerate(selected[normal_regions.cells > 0])}
    a_to_n = {a: original_to_n[original] for a, original in enumerate(originals) if original in original_to_n}
    plan = plan.bind_n_rows(a_to_n)
    n_images, n_masks, replay_records = [], [], []
    if decode_n:
        for slot in active_slots:
            entry = slots[slot]
            trusted = entry["trusted"]
            trace = batch["j2_replay"][slot]
            image = assets.read(context.index.asset_path(trusted, "n"), trusted["n_sha256"], "N")
            replayed = replay_normal(image, entry["combined"], trace)
            require(torch.equal(replayed["masks"][0], batch["masks"][slot]),
                    "Sparse N replay changed native ROI mask identities")
            if materialize_n:
                n_images.append(replayed["img"])
                n_masks.append(replayed["masks"][0])
            replay_records.append({"a_slot": slot, "n_slot": compact[slot], "source_id": trusted["source_id"],
                "n_sha256": trusted["n_sha256"], "source_sha256": trusted["a_sha256"],
                "matrix_sha256": trusted["matrix_identity_sha256"],
                "source_valid_sha256": trusted["source_valid_mask_sha256"],
                "n_valid_sha256": trusted["n_valid_mask_sha256"], "trace_sha256": trace.sha256,
                "A_uint8_tensor": a_image_summaries[slot],
                "N_uint8_tensor": tensor_summary(replayed["img"]),
                "N_float32_tensor": tensor_summary(replayed["img"].float() / 255.),
                "mask_sha256": array_sha(replayed["masks"][0]), "N_native_replays_this_slot": 1})
            if not materialize_n:
                del replayed, image
    normal_images = torch.stack(n_images).float() / 255. if n_images else None
    normal_masks = torch.stack(n_masks) if n_masks else None
    zero_rows = []
    for original in np.flatnonzero(regions.cells == 0):
        slot, ordinal = int(regions.slots[original]), int(regions.ordinals[original])
        native = batch["j2_replay"][slot].native_target_rows[ordinal]
        zero_rows.append({"source_id": native["source_id"], "group_id": native["group_id"],
            "polygon_id": native["polygon_id"], "class_id": int(native["class_id"]),
            "slot": slot, "native_post_format_ordinal": ordinal, "p3_cells": 0,
            "a_pool_row": None, "reason": "ZERO_NATIVE_P3_CELLS"})
    coverage = {"native_poolable_rows": len(rows), "native_zero_cell_rows": int((regions.cells == 0).sum()),
        "native_region_rows": rows, "native_zero_cell_regions": zero_rows,
        "candidate_counts": dict(collection.counts), "matched_rows": len(plan.a_rows),
        "matched_slots": len(active_slots), "N_replays": len(replay_records),
        "N_batch_materialized": bool(materialize_n and n_images),
        "native_geometry": geometry_records,
        "selection": "ALL_ADMITTED_SUPPORT_COMPLETE_NO_B_THEN_CLOSED_MATCHING"}
    require(len(replay_records) == len(active_slots) if decode_n else not replay_records,
            "Every selected N slot must be replayed exactly once")
    require(cpu_rng_equal(before, cpu_rng_snapshot()), "Auxiliary replay/matching consumed native main RNG")
    return PreparedExposure(batch, regions, normal_images, normal_masks, normal_regions,
                            plan, collection, coverage, tuple(rows), tuple(replay_records))


def _exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_record(stream, value):
    stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False).encode() + b"\n")


def committed_epochs(directory, bindings_sha256):
    """Verify immutable completed chunks; uncommitted orphan files are ignored."""
    commits = []
    directory = Path(directory)
    if not directory.exists():
        return commits
    for path in sorted(directory.glob("epoch_*.commit.json")):
        commit = json.loads(path.read_text(encoding="utf-8"))
        require(commit["epoch"] == len(commits) and commit["bindings_sha256"] == bindings_sha256,
                "Epoch commit order or protocol/code binding differs")
        require(Path(commit["chunk"]).name == commit["chunk"], "Unsafe epoch chunk member")
        chunk = directory / commit["chunk"]
        require(not chunk.is_symlink() and sha256(chunk) == commit["chunk_sha256"], "Committed chunk changed")
        with gzip.open(chunk, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        require(len(rows) == 21 and [r["native"]["step"] for r in rows]
                == list(range(commit["epoch"] * 21 + 1, commit["epoch"] * 21 + 22)),
                "Committed epoch batch domain changed")
        commits.append(commit)
    return commits


def replay_source_manifest():
    """Bind executed helpers and pinned native data code, not future GPU stubs."""
    import importlib.metadata
    root = Path(__file__).resolve().parents[2]
    names = ["tools/exp18/" + name for name in ("data.py", "replay.py", "pairing.py", "geometry.py",
             "common.py", "guards.py", "protocol.py", "schedule.py")]
    names += ["tools/exp17/engineering/augmentation.py", "tools/exp17/engineering/data.py",
              "tools/exp15/regions.py", "tools/exp14/audit_admission.py"]
    native = importlib.metadata.distribution("ultralytics")
    package_names = ("data/augment.py", "data/base.py", "data/dataset.py", "data/build.py",
                     "data/utils.py", "utils/instance.py", "cfg/__init__.py")
    return {"repository_files": {name: sha256(root / name) for name in names},
            "native_ultralytics": {name: sha256(native.locate_file("ultralytics/" + name))
                                    for name in package_names},
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "ultralytics", "numpy", "scipy")}}


def run_replay(config, protocol, output, *, max_epochs=None, resume=False, progress=None):
    """Execute model-free full-H data audit; resumptions skip verified epochs."""
    import torch
    from .geometry import native_polygon_area_fraction
    from .protocol import validate_protocol
    protocol = validate_protocol(protocol)
    horizon = protocol["train"]["epochs"]
    target = horizon if max_epochs is None else max_epochs
    require(type(target) is int and 1 <= target <= horizon, "Replay epoch limit exceeds protocol")
    require(not torch.cuda.is_initialized(), "CPU replay requires CUDA uninitialized")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    epochs_dir = output / "epochs"
    epochs_dir.mkdir(exist_ok=resume)
    code = replay_source_manifest()
    binding = {"protocol_sha256": digest(protocol), "runtime_config_sha256": config["runtime_config_sha256"],
               "code": code, "scope": "NATIVE_CPU_DATA_NO_REAL_MODEL_ALL_HORIZON"}
    binding_sha = digest(binding)
    commits = committed_epochs(epochs_dir, binding_sha)
    require(resume or not commits, "Existing replay requires explicit resume")
    references, r5_references = load_references(config)
    attempt = output / ("attempt_" + uuid.uuid4().hex)
    attempt.mkdir()
    _exclusive_json(attempt / "bindings.json", binding)
    started = time.monotonic()
    context = build_dataset(config, attempt / "dataset", horizon=horizon)
    assets = DecodedAssets()
    native_geometry = {label["polygon_id"]: {"source_area_fraction": native_polygon_area_fraction(
        label["normalized_native_coordinates"])} for source in context.index.train.values()
        for label in source["native_labels"]}
    before_release = cpu_rng_snapshot()
    fork_memory = prepare_fork_memory()
    first_release = fork_memory["before_freeze_release"]
    require(cpu_rng_equal(before_release, cpu_rng_snapshot()), "Memory housekeeping consumed native RNG")
    release_calls = 2
    trim_releases = sum(int(value["malloc_trim_return"] == 1)
                        for value in (first_release, fork_memory["after_freeze_release"]))
    try:
        for epoch in range(len(commits), target):
            # The previous auxiliary phase's decoded RAM images are not needed
            # by native workers. Keep their verified identities/counters while
            # releasing arrays before the next fork; no experiment file moves.
            before_release = cpu_rng_snapshot()
            assets.items.clear()
            assets.bytes = 0
            released = release_cpu_memory()
            require(cpu_rng_equal(before_release, cpu_rng_snapshot()),
                    "Decoded RAM cache release consumed native RNG")
            release_calls += 1
            trim_releases += int(released["malloc_trim_return"] == 1)
            chunk = epochs_dir / (f"epoch_{epoch:03d}." + uuid.uuid4().hex + ".jsonl.gz")
            counts = Counter()
            seen = set()
            # Native phase: observe the actual original image tensors, then
            # retain only bounded ROI metadata. No N/geometry/matching is run
            # while the two native workers are alive.
            native_views = []
            loader = new_epoch_loader(context.dataset, epoch, horizon=horizon)
            iterator = iter(loader)
            for batch_index, transported in enumerate(iterator):
                require(batch_index < 21, "Native loader delivered too many batches")
                batch = restore_native_batch(transported)
                actual = native_evidence(context, batch, epoch=epoch, batch_index=batch_index)
                paths = {str(Path(p).resolve(strict=True)) for p in batch["im_file"]}
                require(not seen.intersection(paths), "Within-epoch TRAIN source repetition")
                seen.update(paths)
                historical = compare_historical(actual, batch["j2_replay"], references, r5_references)
                view = retain_native_roi_view(batch, context.groups)
                native_views.append({"native": actual, "historical": historical, "view": view})
                counts.update(batches=1, images=actual["images"], raw_targets=actual["raw_targets"],
                              historical_verified=int(historical))
                del batch, transported, view
                before_release = cpu_rng_snapshot()
                released = release_cpu_memory()
                require(cpu_rng_equal(before_release, cpu_rng_snapshot()),
                        "Native phase memory housekeeping consumed native RNG")
                release_calls += 1
                trim_releases += int(released["malloc_trim_return"] == 1)
                if progress is not None and (batch_index + 1) % 7 == 0:
                    progress({"event": "CPU_NATIVE_BATCH_EVIDENCE_CAPTURED_IN_MEMORY", "epoch": epoch + 1,
                              "batches": batch_index + 1, "committed": False})
            workers = tuple(getattr(iterator, "_workers", ()))
            worker_exit = [{"pid": worker.pid, "exitcode": worker.exitcode} for worker in workers]
            require(len(workers) == 2 and all(not worker.is_alive() and worker.exitcode == 0 for worker in workers),
                    "Auxiliary phase requires both original native workers fully joined")
            require(len(native_views) == counts["batches"] == 21 and counts["images"] == 668
                    and counts["raw_targets"] == 1266 and seen == set(context.groups),
                    "Complete native epoch TRAIN membership/exposure differs")
            del iterator, loader, workers
            before_release = cpu_rng_snapshot()
            released = release_cpu_memory()
            require(cpu_rng_equal(before_release, cpu_rng_snapshot()),
                    "Worker-join memory housekeeping consumed native RNG")
            release_calls += 1
            trim_releases += int(released["malloc_trim_return"] == 1)
            if progress is not None:
                progress({"event": "CPU_NATIVE_EPOCH_CAPTURED_WORKERS_JOINED", "epoch": epoch + 1,
                          "native_workers": worker_exit, "auxiliary_batches_started": 0})
            # Auxiliary phase: no native worker remains. Geometry, closed
            # matching and the real N replays consume original ROI metadata.
            with chunk.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as stream:
                    for batch_index, record in enumerate(native_views):
                        view = record["view"]
                        check_retained_roi_view(view, context.groups)
                        prepared = prepare_exposure(view.batch, context, protocol["matching"], epoch=epoch,
                            batch_index=batch_index, decode_n=True, asset_cache=assets,
                            native_geometry=native_geometry, materialize_n=False,
                            precomputed_a_regions=view.regions, image_shape=view.image_shape,
                            a_image_summaries=view.a_image_summaries)
                        check_retained_roi_view(view, context.groups)
                        counts.update(candidates=len(prepared.collection.candidates),
                                      matched_rows=len(prepared.plan.a_rows), N_replays=len(prepared.replay_records))
                        counts.update({"candidate_" + k: v for k, v in prepared.collection.counts.items()})
                        _write_record(stream, {"schema": "EXP18_CPU_NATIVE_BATCH_V1", "native": record["native"],
                            "six_tensors": view.six_tensors,
                            "native_evidence_scope": {"six_tensors": "REAL_NATIVE_BATCH_BEFORE_IMAGE_RELEASE",
                                "post_aux_rehashed": list(view.retained_tensors),
                                "image_and_semantic_pixels_retained": False,
                                "pre_drop_evidence_sha256": view.evidence_sha256},
                            "historical_J1_R5_data_equal": record["historical"],
                            "traces": [trace.compact_record() for trace in view.batch["j2_replay"]],
                            "coverage": prepared.coverage, "candidates": prepared.collection.to_dict(),
                            "matching": prepared.plan.to_dict(), "N_replay": list(prepared.replay_records)})
                        native_views[batch_index] = None
                        del prepared, view, record
                        before_release = cpu_rng_snapshot()
                        released = release_cpu_memory()
                        require(cpu_rng_equal(before_release, cpu_rng_snapshot()),
                                "Batch memory housekeeping consumed native RNG")
                        release_calls += 1
                        trim_releases += int(released["malloc_trim_return"] == 1)
                raw.flush()
                os.fsync(raw.fileno())
            commit = {"schema": "EXP18_CPU_EPOCH_COMMIT_V1", "epoch": epoch,
                "bindings_sha256": binding_sha, "chunk": chunk.name, "chunk_sha256": sha256(chunk),
                "bytes": chunk.stat().st_size, "counts": dict(counts),
                "native_workers_joined_before_auxiliary": worker_exit}
            _exclusive_json(epochs_dir / f"epoch_{epoch:03d}.commit.json", commit)
            commits.append(commit)
            if progress is not None:
                progress({"event": "CPU_REPLAY_EPOCH_COMMITTED", "epoch": epoch + 1,
                          "horizon": horizon, "counts": dict(counts), "elapsed_seconds": time.monotonic() - started})
        context.postcheck()
        assets.postcheck()
        require(not torch.cuda.is_initialized(), "CPU replay initialized CUDA")
        total = Counter()
        for commit in commits:
            total.update(commit["counts"])
        complete = len(commits) == horizon
        result = {"schema": "EXP18_CPU_REPLAY_SUMMARY_V1",
            "status": "CPU_DATA_READY_GPU_PENDING" if complete else "CPU_REPLAY_PARTIAL",
            "bindings": binding, "bindings_sha256": binding_sha, "horizon": horizon,
            "completed_epochs": len(commits), "counts": dict(total), "commits": commits,
            "elapsed_seconds": time.monotonic() - started, "real_model_constructions": 0,
            "real_model_forwards": 0, "real_checkpoint_tensor_loads": 0, "CUDA_initialized": False,
            "VAL_passes": 0, "TEST_accesses": 0, "lambda_state": None,
            "image_epoch_cache_files": 0, "source_postcheck": "PASS"}
        result["CPU_memory_policy"] = {"decoded_asset_cache_bytes": assets.max_bytes,
            "decoded_asset_RAM_cache_cleared_before_native_phase": True,
            "native_worker_image_buffer": 8, "workers": 2, "prefetch_factor": 1,
            "IPC_transport": "NATIVE_TORCH_FILE_DESCRIPTOR_SHARED_STORAGE_NO_NUMPY_PICKLE_COPY",
            "epoch_stages": ["NATIVE_21_BATCH_CAPTURE_DROP_IMAGE_PIXELS_JOIN_2_WORKERS",
                             "AUXILIARY_GEOMETRY_MATCHING_N_REPLAY_WITH_NO_NATIVE_WORKERS"],
            "retained_epoch_images": 0, "retained_epoch_semantic_masks": 0,
            "original_six_tensor_hash_scope": "REAL_NATIVE_BATCH_BEFORE_PIXEL_RELEASE",
            "N_batch_materialized": False,
            "R5_reference_projection": "NATIVE_AND_TRACES_ONLY_AFTER_COMPLETE_ORIGINAL_FILE_SHA_CHECK",
            "memory_release": "GC_COLLECT_AND_POSIX_LIBC_MALLOC_TRIM_BEFORE_FIRST_FORK_AND_AFTER_EACH_BATCH",
            "fork_memory": fork_memory,
            "first_release": first_release, "release_calls": release_calls,
            "malloc_trim_returned_1_count": trim_releases,
            "native_first_2100_data_and_trace_byte_verification_required": True}
        _exclusive_json(attempt / "summary.json", result)
        return result
    except BaseException as error:
        _exclusive_json(attempt / "failure.json", {"status": "HOLD", "error": repr(error),
            "completed_epochs": len(commits), "preserve_all_partial_artifacts": True})
        raise
