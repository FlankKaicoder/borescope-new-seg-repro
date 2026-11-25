"""Future native three-arm backend; import and refusal paths are model-free.

Nothing in this module constructs a model, loads weights, starts a worker or
initializes CUDA at import. The current EXP18_CPU protocol refuses construction
before importing Torch. A later independent native protocol must additionally
approve lambda, AMP smoke, source/runtime identity and append-only capacity.

CPU preparation lives in a clean child process: the frozen CPU dataset/replay
helpers deliberately reject an already initialized CUDA process. Every epoch
packet and full recovery checkpoint is retained; there is no cache retirement.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

from .common import require, sha256, write_json_exclusive

COCO_SHA256 = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
INITIAL_NATIVE = "8e0479ecdae6dc60558ed44d5d62c01f8b61075514bc34c109295950603fc886"
INITIAL_PROJECTION = "83664f93724017b43a2861826141fc41074a0ae58729d170829f787395d1fc6e"
PACKET_SCHEMA = "EXP18_NATIVE_TRAIN_PREPARED_CPU_V1"


class RetainedRunLease:
    """One OS-backed execution owner; the lock file is never removed/replaced."""
    def __init__(self, path, bindings, *, create=False):
        require(os.name == "posix", "Native run leases require Linux flock")
        import fcntl
        self.path, self.stream, self.held = Path(path).absolute(), None, False
        require(not self.path.is_symlink(), "Run lease is a symlink")
        expected = {"schema": "EXP18_RETAINED_RUN_LEASE_V1", "bindings": bindings}
        try:
            if create:
                self.stream = self.path.open("x+b")
            else:
                require(self.path.is_file(), "Existing run lease file is missing")
                self.stream = self.path.open("rb")
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another process/FD owns this native experiment lease") from exc
            self.held = True
            if create:
                self.stream.write((json.dumps(expected, sort_keys=True, allow_nan=False) + "\n").encode())
                self.stream.flush(); os.fsync(self.stream.fileno())
            else:
                require(json.loads(self.stream.read()) == expected, "Retained run lease binding differs")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.stream is not None:
            if self.held:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream, self.held = None, False

    def __enter__(self):
        require(self.held, "Run lease has already been released")
        return self

    def __exit__(self, *_exc):
        self.close()


def require_native(protocol):
    # This MUST precede every Torch/native import in an execution entry.
    from .protocol import require_operation
    require_operation("train", protocol)
    require(protocol.get("phase") == "EXP18_FORMAL" and protocol.get("formal_training_authorized") is True,
            "Independent native formal authority required")
    coefficient = protocol.get("lambda_state")
    require(type(coefficient) in (int, float) and math.isfinite(coefficient) and coefficient > 0,
            "State coefficient has not passed its independent calibration gate")
    require(protocol["storage"].get("formal_storage_gate") == "PASS",
            "Append-only native artifact capacity gate is pending")


def check_append_only_capacity(output, protocol, resource):
    """Budget ALL epoch packs and ALL immutable step/aux recovery states."""
    epochs, updates = protocol["train"]["epochs"], protocol["train"]["updates"]
    for name in ("epoch_pack_bytes", "recovery_bytes", "reserve_bytes"):
        require(type(resource.get(name)) is int and resource[name] > 0, "Missing measured storage bound: " + name)
    required = (epochs * resource["epoch_pack_bytes"]
                + (updates + epochs + 1) * resource["recovery_bytes"] + resource["reserve_bytes"])
    root = Path(output).resolve()
    probe = root
    while not probe.exists():
        probe = probe.parent
    require(shutil.disk_usage(probe).free >= required,
            f"Append-only full-run storage insufficient: requires {required} bytes; no artifacts may be removed")
    return required


def _legacy_state_hash(state):
    """Original J1 initialization hash, preserving mapping iteration order."""
    import torch
    value = hashlib.sha256()
    for name, tensor in state.items():
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        value.update(name.encode()); value.update(hashlib.sha256(raw).hexdigest().encode())
    return value.hexdigest()


def _replay_reference(config, protocol, epoch):
    """Read the complete CPU audit, including the newly frozen epochs101..200."""
    summary_path = Path(config["replay_summary"])
    require(sha256(summary_path) == config["replay_summary_sha256"], "Full-horizon CPU replay summary changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    horizon = protocol["train"]["epochs"]
    require(summary["status"] == "CPU_DATA_READY_GPU_PENDING" and summary["horizon"] == horizon
            and summary["completed_epochs"] == horizon and len(summary["commits"]) == horizon,
            "Complete H100/H200 CPU data audit is required before any native arm")
    commit = summary["commits"][epoch]
    require(commit["epoch"] == epoch and Path(commit["chunk"]).name == commit["chunk"], "Unsafe replay chunk")
    chunk = Path(config["replay_epochs_root"]) / commit["chunk"]
    require(not chunk.is_symlink() and chunk.stat().st_size == commit["bytes"]
            and sha256(chunk) == commit["chunk_sha256"], "Frozen full-horizon CPU replay chunk changed")
    with gzip.open(chunk, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    require(len(rows) == 21 and [row["native"]["step"] for row in rows] == list(range(epoch * 21 + 1, epoch * 21 + 22)),
            "Frozen full-horizon CPU replay batch domain changed")
    return rows


def _prepare_epoch(spec):
    """Child-only CPU preparation, retaining the original scientific helpers."""
    require_native(spec["protocol"])
    root = Path(spec["directory"]).resolve(strict=True)
    from .guards import configure_cpu_environment
    # In particular, retain native Torch FD resource-sharer socket paths under
    # this owned epoch; no worker may register a filesystem cleanup finalizer.
    configure_cpu_environment(root)
    import torch
    from .data import (build_dataset, compare_historical, load_references, native_evidence,
                       new_epoch_loader, prepare_fork_memory, release_cpu_memory, restore_native_batch)
    from .replay import DecodedAssets, prepare_exposure
    from .checkpoint import state_hash

    require(not torch.cuda.is_initialized(), "Prepared-packet child inherited CUDA")
    protocol, config, epoch = spec["protocol"], spec["config"], spec["epoch"]
    context = build_dataset(config, root / "dataset", horizon=protocol["train"]["epochs"])
    reference, r5_reference = load_references(config)
    exp18_reference = _replay_reference(config, protocol, epoch)
    assets = DecodedAssets(max_bytes=32 * 1024**2)
    write_json_exclusive(root / "fork_memory.json", prepare_fork_memory())
    rows = []
    for batch_index, transported in enumerate(new_epoch_loader(context.dataset, epoch,
            seed=protocol["seed"], horizon=protocol["train"]["epochs"])):
        require(batch_index < 21, "Native epoch delivered excess batches")
        batch = restore_native_batch(transported)
        native = native_evidence(context, batch, epoch=epoch, batch_index=batch_index)
        compare_historical(native, batch["j2_replay"], reference, r5_reference)
        prepared = prepare_exposure(batch, context, protocol["matching"], epoch=epoch,
            batch_index=batch_index, decode_n=spec["arm"] != "j1", asset_cache=assets,
            materialize_n=True)
        require(native_evidence(context, batch, epoch=epoch, batch_index=batch_index) == native,
                "CPU auxiliary preparation changed native batch bytes")
        frozen = exp18_reference[batch_index]
        require(native == frozen["native"] and prepared.plan.to_dict() == frozen["matching"]
                and [trace.compact_record() for trace in batch["j2_replay"]] == frozen["traces"],
                "Native prepared exposure differs from the frozen full-horizon CPU audit")
        if spec["arm"] != "j1":
            require(list(prepared.replay_records) == frozen["N_replay"], "N replay bytes differ from the full-horizon CPU audit")
        packet = {"schema": PACKET_SCHEMA, "bindings": spec["bindings"], "epoch": epoch,
                  "batch_index": batch_index, "native": native, "prepared": prepared,
                  "matching_sha256": prepared.plan.sha256}
        path = root / f"batch_{batch_index:03d}.pt"
        with path.open("xb") as stream:
            torch.save(packet, stream); stream.flush(); os.fsync(stream.fileno())
        rows.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                     "native": native, "matching_sha256": prepared.plan.sha256})
        # Do not retain the previous FP32 N batch while materializing the next.
        packet = prepared = batch = transported = None
        release_cpu_memory()
    require(len(rows) == 21, "Native epoch delivered incomplete batches")
    context.postcheck(); assets.postcheck()
    manifest = {"schema": PACKET_SCHEMA, "bindings": spec["bindings"], "epoch": epoch,
                "batches": rows, "cpu_only": True, "data_configuration_sha256": state_hash(config)}
    write_json_exclusive(root / "manifest.json", manifest)


class PreparedCursor:
    """Only own, fully sealed CPU packets can enter the GPU main process."""
    def __init__(self, directory, bindings, *, manifest_sha256, next_batch=0):
        self.directory = Path(directory).resolve(strict=True)
        require(not Path(directory).is_symlink(), "Epoch packet directory is a symlink")
        path = self.directory / "manifest.json"
        require(not path.is_symlink() and sha256(path) == manifest_sha256, "Epoch packet manifest changed")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        require(self.manifest["schema"] == PACKET_SCHEMA and self.manifest["bindings"] == bindings
                and self.manifest["cpu_only"] is True and len(self.manifest["batches"]) == 21,
                "Epoch packet schema/binding differs")
        require(type(next_batch) is int and 0 <= next_batch <= 21, "Invalid epoch packet cursor")
        self.bindings, self.manifest_sha256, self.next_index = copy.deepcopy(bindings), manifest_sha256, next_batch

    def state_dict(self):
        return {"schema": PACKET_SCHEMA, "epoch": self.manifest["epoch"], "next_batch": self.next_index,
                "directory": str(self.directory), "cache_manifest_sha256": self.manifest_sha256}

    def next_batch(self):
        require(self.next_index < 21, "Epoch packet cursor exhausted")
        manifest = self.directory / "manifest.json"
        require(sha256(manifest) == self.manifest_sha256, "Sealed epoch manifest changed while consuming")
        row = self.manifest["batches"][self.next_index]
        require(row["file"] == f"batch_{self.next_index:03d}.pt", "Unsafe packet member")
        path = self.directory / row["file"]
        require(not path.is_symlink() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                "Prepared CPU packet byte identity changed")
        import torch
        # Trusted private dataclasses are allowed only after this run's full SHA binding.
        packet = torch.load(path, map_location="cpu", weights_only=False)
        require(packet["schema"] == PACKET_SCHEMA and packet["bindings"] == self.bindings
                and packet["epoch"] == self.manifest["epoch"] and packet["batch_index"] == self.next_index
                and packet["native"] == row["native"] and packet["matching_sha256"] == row["matching_sha256"],
                "Prepared CPU packet metadata differs from sealed manifest")
        require(packet["prepared"].plan.sha256 == packet["matching_sha256"], "Matching plan hash changed")
        self.next_index += 1
        return packet


def native_trainer_factory(context, config, protocol, output, arm, *, lease=None):
    """Construct after explicit gates; heads precede native optimizer and EMA."""
    require_native(protocol)
    require(isinstance(lease, RetainedRunLease) and lease.held
            and lease.path.parent.resolve() == Path(output).resolve().parent,
            "Native model setup requires this arm's already-held execution lease")
    import torch
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from tools.exp15.regions import add_projection
    from .data import AUGMENTATION, CLASSES, new_epoch_loader
    from .model import add_state_head

    train = protocol["train"]
    namespace = protocol["method"]["state_head"]["seed_namespace"]
    head_seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "little")
    overrides = {"model": config["weight"], "data": "EXP18_FROZEN_TRAIN_ONLY", "task": "segment",
        "device": "0", "seed": protocol["seed"], "epochs": train["epochs"], "batch": 32,
        "nbs": 32, "imgsz": 640, "workers": 2, "optimizer": "AdamW", "lr0": .01, "lrf": .01,
        "momentum": .937, "weight_decay": .0005, "cos_lr": False, "warmup_epochs": 3.,
        "warmup_bias_lr": .1, "warmup_momentum": .8, "amp": True, "deterministic": True,
        "compile": False, "patience": 0, "val": False, "plots": False, "save": False,
        "resume": False, "cache": False, "fraction": 1., "single_cls": False,
        "overlap_mask": True, "mask_ratio": 4, "project": str(output), "name": "runtime",
        "exist_ok": False, **AUGMENTATION}

    class Exp18Trainer(SegmentationTrainer):
        def get_dataset(self):
            return {"train": config["train_list"], "val": None, "test": None, "nc": 7,
                    "channels": 3, "names": dict(enumerate(CLASSES))}

        def get_model(self, cfg=None, weights=None, verbose=True):
            require(weights is not None and len(weights.names) == 80, "Fresh frozen COCO initialization required")
            model = super().get_model(cfg, weights, verbose)
            source, current = weights.state_dict(), model.state_dict()
            matched = [name for name, value in source.items() if name in current and current[name].shape == value.shape]
            require(matched and all(torch.equal(current[name], source[name]) for name in matched)
                    and model.model[-1].nc == 7, "COCO/native seven-class transfer changed")
            add_projection(model)
            if arm != "j1":
                add_state_head(model, seed=head_seed)
            return model

        def build_dataset(self, img_path, mode="train", batch=None):
            require(mode == "train" and img_path == self.data["train"] and batch == 32,
                    "Only frozen native TRAIN is allowed")
            return context.dataset

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "val":
                require(dataset_path is None, "Held-out dataset requested during TRAIN setup")
                return None
            require(batch_size == 32 and rank == -1, "Single-device batch32 TRAIN required")
            # An ordinary lazy DataLoader supplies setup metadata. Epoch packets
            # are made by a separate clean CPU process, never iterated here.
            return new_epoch_loader(context.dataset, 0, seed=protocol["seed"],
                                    horizon=train["epochs"], numpy_transport=False)

        def get_validator(self):
            return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

        def run_callbacks(self, _name):
            pass

        def train(self, *_args, **_kwargs):
            raise RuntimeError("Use the independently gated EXP18 transaction executor")

        def validate(self):
            raise RuntimeError("Only the injected isolated EXP18 VAL evaluator may run")

        def final_eval(self):
            raise RuntimeError("Final evaluation belongs to the explicit EXP18 VAL scope")

    with patch("ultralytics.engine.trainer.callbacks.add_integration_callbacks"):
        trainer = Exp18Trainer(overrides=overrides)
    with patch("ultralytics.engine.trainer.check_amp", return_value=True):
        trainer._setup_train()
    require(trainer.device.type == "cuda" and trainer.world_size == 1 and not trainer.ddp
            and trainer.accumulate == 1 and len(trainer.train_loader) == 21
            and trainer.test_loader is None and trainer.amp, "Native execution topology/configuration changed")
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=1.)
    trainer.optimizer.zero_grad(set_to_none=True)
    return trainer


class NativeAdapter:
    """Transaction backend for ExperimentExecutor; current protocol rejects it."""
    def __init__(self, config, protocol, bindings, output, *, arm, evaluator, exporter, diagnostic=None, resume=False):
        require_native(protocol)
        require(arm in protocol["arms"] and callable(evaluator) and callable(exporter),
                "Arm and concrete evaluator/exporter callbacks are required")
        requested_output = Path(output).absolute()
        require(not requested_output.is_symlink(), "Native output root is a symlink")
        self.output = requested_output.resolve()
        for protected in (Path("/root/autodl-tmp/损伤训练数据集"), Path(config["release"])):
            protected = protected.resolve()
            require(not self.output.is_relative_to(protected) and not protected.is_relative_to(self.output),
                    "Native output intersects permanently read-only raw data or frozen release")
        check_append_only_capacity(self.output, protocol, config["resources"])
        require(sha256(config["weight"]) == COCO_SHA256, "Frozen COCO checkpoint bytes changed")
        require(bindings.get("replay_summary_sha256") == config["replay_summary_sha256"],
                "Full-horizon CPU replay is not bound to this native arm")
        _replay_reference(config, protocol, 0)
        self.config, self.protocol, self.bindings, self.arm = copy.deepcopy(config), copy.deepcopy(protocol), copy.deepcopy(bindings), arm
        self.evaluator, self.exporter, self.diagnostic = evaluator, exporter, diagnostic
        owner = {"schema": "EXP18_NATIVE_ARM_OWNER_V1", "arm": arm, "bindings": bindings}
        if resume:
            marker = self.output / "owner.json"
            require(marker.is_file() and not marker.is_symlink()
                    and json.loads(marker.read_text(encoding="utf-8")) == owner, "Native resume owner binding differs")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            write_json_exclusive(self.output / "owner.json", owner)
        self._lease = RetainedRunLease(self.output / "run.lock", bindings, create=not resume)
        try:
            self._initialize_native()
        except BaseException:
            self.close()
            raise

    def _initialize_native(self):
        config, protocol, arm = self.config, self.protocol, self.arm
        import torch
        import torchvision
        import ultralytics
        from .data import build_dataset, load_references
        from .model import native_state_dict
        from .optimizer import HeadAwareEMA
        from .checkpoint import optimizer_names

        require(torch.__version__ == "2.8.0+cu128" and ultralytics.__version__ == "8.4.117",
                "Pinned native package versions changed")
        require(not torch.cuda.is_initialized(), "Construct TRAIN context before CUDA discovery/model setup")
        framework = Path(ultralytics.__file__).parent
        require(config.get("framework_sources_sha256"), "Frozen native source hashes required")
        for name, expected in config["framework_sources_sha256"].items():
            require(sha256(framework / name) == expected, "Pinned native framework source changed: " + name)
        self.context = build_dataset(config, self.output / "dataset", horizon=protocol["train"]["epochs"])
        self.references, self.r5_references = load_references(config)
        # Dataset construction must precede CUDA discovery; runtime comparison
        # must precede any real model construction or COCO tensor loading.
        runtime = {"gpu": torch.cuda.get_device_name(0), "cuda_devices": torch.cuda.device_count(),
                   "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "python": sys.version.split()[0]}
        require(runtime == config["runtime"] and runtime["cuda_devices"] == 1,
                "Runtime differs from the separately measured single-GPU native smoke environment")
        number = 0
        setup = self.output / f"setup_{number:04d}"
        while setup.exists():
            number += 1
            setup = self.output / f"setup_{number:04d}"
        setup.mkdir(exist_ok=False)
        self.trainer = native_trainer_factory(self.context, config, protocol, setup, arm, lease=self._lease)
        self.model, self.optimizer, self.scheduler, self.scaler = (self.trainer.model, self.trainer.optimizer,
                                                               self.trainer.scheduler, self.trainer.scaler)
        self.ema = HeadAwareEMA(self.trainer.ema)
        self.trainer.ema = self.ema
        self.optimizer_names = optimizer_names(self.model, self.optimizer)
        require([g["param_group"] for g in self.optimizer.param_groups] == ["weight", "bn", "bias"],
                "Native AdamW group order changed")
        require(_legacy_state_hash(native_state_dict(self.model)) == INITIAL_NATIVE,
                "Fresh native seed42 initialization differs from original J1")
        projection = {n: p for n, p in self.model.named_parameters() if n.startswith("region_projection.")}
        require(_legacy_state_hash(projection) == INITIAL_PROJECTION, "J1 projection initialization changed")
        head_names = [n for group in self.optimizer_names for n in group if n.startswith("state_head.")]
        require(head_names == ([] if arm == "j1" else ["state_head.weight"]), "State-head optimizer membership differs")
        self.counters = {"optimizer_attempts": 0, "amp_skips": 0}
        self.current_epoch, self.last_matching, self.last_packet = -1, None, None

    def start_epoch(self, epoch):
        require(self._lease.held, "Native execution lease has been released")
        require(0 <= epoch < self.protocol["train"]["epochs"] and self.counters["optimizer_attempts"] == epoch * 21
                and self.scheduler.last_epoch == epoch - 1, "Epoch scheduler/optimizer boundary differs")
        self.current_epoch = self.trainer.epoch = epoch
        self.trainer._model_train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.scheduler.step()

    def build_cache(self, epoch):
        require_native(self.protocol)
        require(epoch == self.current_epoch, "Cache must follow native epoch start")
        root = self.output / "epoch_cache" / f"epoch_{epoch:03d}"
        root.mkdir(parents=True, exist_ok=False)
        spec = {"protocol": self.protocol, "config": self.config, "bindings": self.bindings,
                "arm": self.arm, "directory": str(root), "epoch": epoch}
        spec_path = root / "spec.json"
        write_json_exclusive(spec_path, spec)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        with (root / "worker.log").open("xb") as log:
            result = subprocess.run([sys.executable, "-B", "-m", "tools.exp18.native_adapter", "prepare-epoch", str(spec_path)],
                cwd=Path(__file__).resolve().parents[2], env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        require(result.returncode == 0, "Clean CPU epoch preparation failed; preserve its complete log/partial artifacts")
        manifest = root / "manifest.json"
        cursor = PreparedCursor(root, self.bindings, manifest_sha256=sha256(manifest))
        size = sum(row["bytes"] for row in cursor.manifest["batches"])
        require(size <= self.config["resources"]["epoch_pack_bytes"], "Epoch cache exceeds measured storage gate")
        return cursor

    def open_cache(self, sampler):
        directory = Path(sampler["directory"]).resolve(strict=True)
        require(directory.parent == (self.output / "epoch_cache").resolve(strict=True), "Recovery epoch cache escaped this run")
        self.current_epoch = self.trainer.epoch = sampler["epoch"]
        require(self.scheduler.last_epoch == self.current_epoch, "Recovered scheduler differs from sealed epoch")
        return PreparedCursor(directory, self.bindings, manifest_sha256=sampler["cache_manifest_sha256"],
                              next_batch=sampler["next_batch"])

    def before_batch(self, step):
        from .schedule import point
        require(self._lease.held, "Native execution lease has been released")
        require(step == self.counters["optimizer_attempts"] + 1 and (step - 1) // 21 == self.current_epoch,
                "Native update skipped/repeated or left its epoch")
        require(self.scheduler.last_epoch == self.current_epoch, "Native scheduler epoch differs")
        resource = self.config["resources"]
        require(shutil.disk_usage(self.output).free >= resource["recovery_bytes"] + resource["reserve_bytes"],
                "Insufficient immutable checkpoint headroom before optimizer update")
        expected = point(step, self.protocol["train"]["epochs"])
        for group, lr, decay in zip(self.optimizer.param_groups, expected["lr"], expected["weight_decay"], strict=True):
            require(list(group["betas"]) == expected["betas"] and group["weight_decay"] == decay
                    and group["eps"] == 1e-8 and "momentum" not in group, "Native AdamW numerical settings changed")
            group["lr"] = lr

    def forward_batch(self, packet, step):
        import torch
        from tools.exp15 import regions as j1
        from tools.exp17.engineering.model import prefix_p3
        from .data import compare_historical, native_evidence
        from .model import AForward, forward_joint

        prepared = packet["prepared"]
        original = native_evidence(self.context, prepared.batch, epoch=self.current_epoch, batch_index=(step - 1) % 21)
        require(original == packet["native"] and original["step"] == step, "Sealed native packet no longer matches TRAIN bytes")
        compare_historical(original, prepared.batch["j2_replay"], self.references, self.r5_references)
        batch = {k: v.to(self.trainer.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in prepared.batch.items()}
        batch["img"] = batch["img"].float() / 255.
        n_images = prepared.normal_images.to(self.trainer.device) if prepared.normal_images is not None else None
        n_masks = prepared.normal_masks.to(self.trainer.device) if prepared.normal_masks is not None else None

        def a_forward(model, native_batch, regions):
            p3 = []
            handle = model.model[16].register_forward_hook(lambda _m, _i, value: p3.append(value))
            try:
                yolo, parts = model(native_batch)
            finally:
                handle.remove()
            require(len(p3) == 1 and p3[0].shape[1] == 64 and p3[0].grad_fn is not None,
                    "Native A must expose one differentiable raw64 fused P3")
            embedding = j1.project_regions(p3[0], native_batch["masks"], regions, model.region_projection)
            supcon = j1.supervised_contrast(embedding, regions.classes, regions)
            return AForward(yolo.sum(), supcon, p3[0], parts)

        with torch.autocast("cuda", enabled=self.trainer.amp):
            result = forward_joint(self.model, batch, prepared.a_regions, n_images, n_masks,
                prepared.normal_regions, prepared.plan, arm=self.arm, lambda_state=self.protocol["lambda_state"],
                a_forward=a_forward, n_forward=prefix_p3)
        self.last_matching, self.last_packet = result.matching_sha256, packet["native"]
        return result

    def after_batch(self, row):
        from .checkpoint import optimizer_names
        require(optimizer_names(self.model, self.optimizer) == self.optimizer_names, "Native parameter group membership changed")
        require(row["global_update"] == self.counters["optimizer_attempts"] + 1,
                "Successful native optimizer step count changed")
        self.counters["optimizer_attempts"] = row["global_update"]

    def capture(self, *, position, sampler_state, head_active_updates, extra):
        from .checkpoint import capture_training_state
        return capture_training_state(self.model, self.optimizer, self.scheduler, self.ema,
            bindings=self.bindings, position=position, sampler_state=sampler_state,
            head_active_updates=head_active_updates, scaler=self.scaler, counters=self.counters,
            extra={**extra, "last_matching_sha256": self.last_matching, "last_native_packet": self.last_packet},
            model_origin="native", native_protocol=self.protocol)

    def restore(self, record):
        from .checkpoint import restore_training_state
        state = restore_training_state(self.model, self.optimizer, self.scheduler, self.ema, record,
            bindings=self.bindings, scaler=self.scaler, native_protocol=self.protocol)
        self.counters = state["counters"]
        self.last_matching = state["extra"].get("last_matching_sha256")
        self.last_packet = state["extra"].get("last_native_packet")
        return state

    def evaluate(self, epoch, role):
        from .validation import run_isolated_native
        return run_isolated_native(self.evaluator, self, epoch=epoch, role=role)

    def diagnostics(self, epoch):
        if self.diagnostic is not None:
            return self.diagnostic(self, epoch=epoch)

    def export(self, role):
        require_native(self.protocol)
        return self.exporter(self, role=role)

    def close(self):
        if getattr(self, "_lease", None) is not None:
            self._lease.close()


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare-epoch",))
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    _prepare_epoch(json.loads(args.spec.read_text(encoding="utf-8")))


if __name__ == "__main__":
    _main()
