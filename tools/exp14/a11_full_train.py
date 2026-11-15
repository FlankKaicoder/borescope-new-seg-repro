"""Prepare, or under a separate explicit gate run, an audited formal E0/E1 training arm."""

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

from tools.exp13_cloud.g3.g3_batch_executor import write_json_exclusive
from tools.exp14.a11_full_protocol import authorize_training, load_protocol, validate_identity, verify_inputs
from tools.exp14.audit_admission import require, sha256_file


APPROVED_RESUME = "EXP14_E0_E1_FULL_RESUME_AUTHORIZATION"
PASS_ARM = "PASS_FULL_TRAIN_ARM"
HARD_GATE = "HARD_GATE"
INTERRUPTED = "INTERRUPTED_REVIEW_REQUIRED"
CHECKPOINT_INTERVAL = 5
MIN_RUN_FREE_BYTES = 2 * 1024**3


def epoch_seed(seed, epoch):
    """Map an experiment seed and zero-based epoch to a portable 64-bit seed."""
    require(isinstance(seed, int) and seed >= 0 and 0 <= epoch < 100, "Invalid formal epoch seed inputs")
    payload = f"exp14-a11-full:v1:{seed}:{epoch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")


def check_epoch_counts(observed, spec, batches, tail, expected, seen):
    require(seen == expected and batches == spec["batches_per_epoch"] and tail == spec["tail_batch"],
            "Formal TRAIN epoch membership, batch count or tail differs")
    expected_exposure = {"original_exposures": spec["original_images"], "repair_exposures": spec["repair_images"],
                         "raw_instance_exposures": spec["raw_instances"]}
    require(all(observed[k] == v for k, v in expected_exposure.items()), "Formal TRAIN per-epoch exposure differs")


def authorize_resume(args, trainer_identity):
    checkpoint = args.resume_checkpoint.resolve(strict=True)
    prior = checkpoint.parent.parent
    require(checkpoint.parent.name == "checkpoints" and prior != args.output.resolve(),
            "Resume requires an earlier independent segment checkpoint")
    available = sorted(checkpoint.parent.glob("epoch_[0-9][0-9][0-9].pt"))
    require(available and checkpoint == available[-1] and not (prior / "last.pt").exists(),
            "Resume requires the latest sealed checkpoint boundary in an unfinished arm")
    events = prior / "events.jsonl"
    summary = prior / "summary.json"
    gate = json.loads(args.resume_gate.read_text(encoding="utf-8"))
    required = {"stage": APPROVED_RESUME, "decision": "PASS", "arm": args.arm,
                "protocol_sha256": trainer_identity["protocol_sha256"], "identity_sha256": trainer_identity["identity_sha256"],
                "training_authorization_sha256": sha256_file(args.apply_gate), "checkpoint_sha256": sha256_file(checkpoint),
                "prior_events_sha256": sha256_file(events)}
    require(gate == required, "Explicit resume Gate does not bind this arm, checkpoint and prior event ledger")
    process = json.loads((prior / "process.json").read_text(encoding="utf-8"))
    if process["host"] == socket.gethostname():
        require(process["linux_process_start"] is None or linux_process_start(process["pid"]) != process["linux_process_start"],
                "The earlier formal segment is still running")
    if summary.is_file():
        previous = json.loads(summary.read_text(encoding="utf-8"))
        require(previous["status"] == INTERRUPTED and previous["arm"] == args.arm,
                "Only non-numeric interrupted formal segments may resume")
    content = events.read_bytes()
    complete = content.split(b"\n")[:-1]
    entries = [json.loads(raw) for raw in complete]
    require(not any(x["event"] in {"forward_stopped", "optimizer_stopped", "hard_gate"} for x in entries),
            "Formal segment contains a numerical or input hard failure")
    return checkpoint, prior, required


def epoch_ledger_entry(event_file, completed_epochs, counters, previous_segments, epoch_batches):
    return {"completed_epochs": completed_epochs, "counters": dict(counters), "segments": previous_segments,
            "epoch_batches": epoch_batches, "target_epochs": 100, "optimizer_updates": counters["batches_read"],
            "epoch_ledger_sha256": sha256_file(event_file), "event_bytes": event_file.stat().st_size}


def restore_ledger(metadata, event_file, expected):
    require(0 < metadata["completed_epochs"] < 100 and metadata["epoch_batches"] == expected["batches_per_epoch"],
            "Resume checkpoint is not an incomplete formal epoch boundary")
    completed = metadata["completed_epochs"]
    updates = completed * metadata["epoch_batches"]
    counters = Counter(metadata["counters"])
    require(metadata["optimizer_updates"] == updates
            and all(counters[k] == updates for k in ("batches_read", "forwards", "backwards", "finite_forwards", "completed_backwards"))
            and counters["complete_epochs"] == completed, "Resume counters do not attest a complete epoch boundary")
    with event_file.open("rb") as f:
        prefix = f.read(metadata["event_bytes"])
    require(len(prefix) == metadata["event_bytes"] and hashlib.sha256(prefix).hexdigest() == metadata["epoch_ledger_sha256"],
            "Resume checkpoint does not bind the original epoch event prefix")
    require(metadata["target_epochs"] == 100
            and counters["original_exposures"] == completed * expected["original_images"]
            and counters["repair_exposures"] == completed * expected["repair_images"]
            and counters["raw_instance_exposures"] == completed * expected["raw_instances"],
            "Resume source or target exposure does not attest the frozen epoch policy")
    return completed, counters


def discarded_ledger(event_file, byte_offset):
    content = event_file.read_bytes()[byte_offset:]
    events = [json.loads(raw) for raw in content.split(b"\n")[:-1]]
    reads = [x for x in events if x["event"] == "batch_read"]
    optimizer_events = [x for x in events if x["event"] == "optimizer"]
    return {"known_discarded_batches": len(reads),
            "known_discarded_forwards": sum(x["event"] == "forward" for x in events),
            "known_discarded_backwards": sum(x["event"] == "backward" for x in events),
            "known_discarded_native_steps": sum(x["native_step_called"] for x in optimizer_events),
            "known_discarded_original_exposures": sum(x["original_images"] for x in reads),
            "known_discarded_repair_exposures": sum(x["repaired_images"] for x in reads),
            "potentially_inflight_batches": len(reads) - len(optimizer_events),
            "trailing_partial_event": not content.endswith(b"\n") if content else False}


def build_formal_trainer():
    # The default preparation path need not import or initialize Torch or Ultralytics.
    import torch
    import torchvision
    import ultralytics
    from torch.utils.data import DataLoader, get_worker_info
    from ultralytics.data.augment import Albumentations
    from ultralytics.data.build import seed_worker
    from ultralytics.utils.torch_utils import autocast

    from tools.exp14.a11_full_numerics import FormalNumericalAudit, load_epoch_checkpoint, save_epoch_checkpoint
    from tools.exp14.a11_smoke_train import BoundedSmokeTrainer

    def init_worker(worker_id):
        seed_worker(worker_id)
        # New Albumentations versions keep an internal RNG in addition to Python/NumPy.
        def transforms(node):
            yield node
            for child in getattr(node, "transforms", ()) or ():
                yield from transforms(child)

        for native in transforms(get_worker_info().dataset.transforms):
            if isinstance(native, Albumentations) and hasattr(native.transform, "set_random_seed"):
                native.transform.set_random_seed(torch.initial_seed())

    class FormalTrainer(BoundedSmokeTrainer):
        def __init__(self, release, weight, output, arm, protocol, identity):
            self.formal = protocol["training"]
            self.expected_arm = protocol["arms"][arm]
            self.identity = identity
            self.runtime = {"gpu": torch.cuda.get_device_name(0), "cuda_devices": torch.cuda.device_count(),
                            "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                            "ultralytics": ultralytics.__version__, "python": sys.version.split()[0]}
            require(self.runtime == protocol["runtime"], "Formal GPU or framework differs from the sealed runtime")
            self.completed_epochs = 0
            self.previous_segments = []
            self.last_checkpoint = None
            super().__init__(release, weight, output, arm, self.formal["batch"])
            self.args.patience = self.formal["patience"]
            self.checkpoint_identity = {**self.checkpoint_identity, **identity,
                                        "updates_cap": self.expected_arm["effective_updates"]}

        def new_epoch_loader(self, dataset, epoch):
            generator = torch.Generator().manual_seed(epoch_seed(self.formal["seed"], epoch))
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.formal["workers"],
                              collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False, persistent_workers=False,
                              generator=generator, worker_init_fn=init_worker, multiprocessing_context="fork")

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "val":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            require(rank == -1 and batch_size == self.formal["batch"], "Formal loader must be single-device and protocol-sized")
            return self.new_epoch_loader(self.build_dataset(dataset_path, mode, batch_size), 0)

        def setup(self):
            super().setup()
            self.audit.close()
            self.audit = FormalNumericalAudit(self, self.expected_arm["effective_updates"])
            expected_hyp = {**self.formal, **self.formal["augmentation"]}
            overrides = ("seed", "epochs", "batch", "nbs", "imgsz", "optimizer", "lr0", "lrf", "cos_lr",
                         "warmup_epochs", "warmup_bias_lr", "warmup_momentum", "momentum", "weight_decay",
                         "amp", "workers", "deterministic", "compile", "patience")
            require(all(getattr(self.args, k) == expected_hyp[k] for k in (*overrides, *self.formal["augmentation"]))
                    and self.accumulate == self.formal["accumulate"]
                    and self.train_loader.drop_last == self.formal["drop_last"]
                    and not self.args.val and not self.formal["validation_during_training"]
                    and self.formal["checkpoint_interval_epochs"] == CHECKPOINT_INTERVAL
                    and self.scaler.get_scale() == self.formal["amp_initial_scale"]
                    and self.scaler.get_growth_interval() == self.formal["amp_growth_interval"],
                    "Effective native trainer arguments differ from the sealed formal protocol")
            require(len(self.expected) == self.expected_arm["images"] and
                    self.expected_arm["batches_per_epoch"] == len(self.train_loader), "Formal dataset or loader count differs")

        def close(self):
            if self.audit:
                self.audit.close()

        def resume_segment(self, resume):
            checkpoint, prior, gate = resume
            metadata = load_epoch_checkpoint(self, checkpoint, self.checkpoint_identity)
            self.completed_epochs, self.counters = restore_ledger(metadata, prior / "events.jsonl", self.expected_arm)
            require(self.audit.effective_updates == self.counters["batches_read"],
                    "Restored optimizer state and epoch ledger disagree")
            self.previous_segments = [*metadata["segments"], {"events_sha256": gate["prior_events_sha256"],
                                      "checkpoint_sha256": gate["checkpoint_sha256"],
                                      "completed_epochs": self.completed_epochs,
                                      "discarded_ledger": discarded_ledger(prior / "events.jsonl", metadata["event_bytes"])}]
            self.event("resume", completed_epochs=self.completed_epochs, segments=self.previous_segments)

        def check_input(self, batch):
            img = batch["img"]
            return (img.device.type == "cuda" and img.dtype == torch.float32 and img.shape[1:] == (3, self.args.imgsz, self.args.imgsz)
                    and bool(((img >= 0) & (img <= 1)).all())
                    and bool(torch.isfinite(batch["bboxes"]).all()) and bool(torch.isfinite(batch["cls"]).all()))

        def train_batch(self, batch):
            number = self.counters["batches_read"]
            with autocast(self.amp, device="cuda"):
                require(torch.is_autocast_enabled("cuda"), "Formal CUDA autocast is inactive")
                native = self.preprocess_batch(batch)
                require(self.check_input(native), "Nonfinite, malformed or out-of-range augmented training input")
                self.counters["forwards"] += 1
                raw_loss, parts = self.model(native)
                self.loss, self.loss_items = raw_loss.sum(), parts
                try:
                    checked = self.audit.check_forward(self.loss, parts)
                except Exception as error:
                    self.event("forward_stopped", batch=number, **getattr(error, "evidence", {}))
                    raise
                self.event("forward", batch=number, **checked)
                self.counters["finite_forwards"] += 1
            self.counters["backwards"] += 1
            self.scaler.scale(self.loss).backward()
            self.counters["completed_backwards"] += 1
            try:
                checked = self.audit.optimizer_step()
            except Exception as error:
                self.event("optimizer_stopped", batch=number, **getattr(error, "evidence", {}))
                raise
            self.event("optimizer", batch=number, **checked)
            require(self.audit.optimizer_attempts == self.audit.effective_updates == number, "Formal effective update budget differs")

        def run_formal(self, resume):
            self.setup()
            if resume:
                self.resume_segment(resume)
            arm, total, nb = self.expected_arm, self.formal["epochs"], len(self.train_loader)
            nw = self._get_warmup_iterations(nb)
            require(nw == arm["warmup_iterations"] and total * nb == arm["effective_updates"], "Formal schedule differs from protocol")
            dataset, expected = self.train_loader.dataset, self.expected.keys()
            for self.epoch in range(self.completed_epochs, total):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.scheduler.step()
                self._model_train()
                observed = self.counters.copy()
                seen, batches, tail = set(), 0, 0
                self.event("epoch_start", epoch=self.epoch + 1, rng_seed=epoch_seed(self.formal["seed"], self.epoch),
                           iteration_offset=self.counters["batches_read"])
                for batch in self.new_epoch_loader(dataset, self.epoch):
                    require(self.counters["batches_read"] < arm["effective_updates"], "Formal batch budget exhausted")
                    self.warmup(self.counters["batches_read"], nw)
                    paths = self.audit_batch(batch)
                    require(not seen.intersection(paths), "A formal epoch repeated a TRAIN image")
                    seen.update(paths)
                    batches += 1
                    tail = len(paths)
                    self.train_batch(batch)
                check_epoch_counts(self.counters - observed, arm, batches, tail, expected, seen)
                self.completed_epochs = self.epoch + 1
                self.counters["complete_epochs"] = self.completed_epochs
                self.event("epoch_complete", epoch=self.completed_epochs, batches=batches, tail_batch=tail,
                           raw_instance_exposures=(self.counters - observed)["raw_instance_exposures"],
                           original_exposures=(self.counters - observed)["original_exposures"],
                           repair_exposures=(self.counters - observed)["repair_exposures"],
                           augmented_instances=(self.counters - observed)["augmented_instances"],
                           optimizer_updates=self.audit.effective_updates, scale=self.scaler.get_scale())
                if self.completed_epochs == total or self.completed_epochs % CHECKPOINT_INTERVAL == 0:
                    path = (self.output / "last.pt" if self.completed_epochs == total else
                            self.output / "checkpoints" / f"epoch_{self.completed_epochs:03d}.pt")
                    path.parent.mkdir(exist_ok=True)
                    report = epoch_ledger_entry(self.events, self.completed_epochs, self.counters, self.previous_segments, nb)
                    self.last_checkpoint = save_epoch_checkpoint(self, path, report)
                    self.event("checkpoint", epoch=self.completed_epochs, sha256=sha256_file(path))
            require(self.completed_epochs == total and self.counters["batches_read"] == self.counters["backwards"]
                    == self.counters["finite_forwards"] == self.audit.native_steps == self.audit.effective_updates
                    == arm["effective_updates"] and self.audit.amp_skips == 0, "Formal successful arm count does not reconcile")
            require(all(self.counters[k] == arm[k] for k in ("original_exposures", "repair_exposures", "raw_instance_exposures")),
                    "Complete 100-epoch source exposure differs")

    return FormalTrainer


def linux_process_start(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    return f"{boot_id}:{stat.rsplit(')', 1)[1].split()[19]}"


def outside_frozen_roots(path, release):
    return not any(path.resolve().is_relative_to(root) for root in
                   (release.resolve(), Path("/root/autodl-tmp/borescope-new-seg-data/v1").resolve(),
                    Path("/root/autodl-tmp/损伤训练数据集").resolve()))


def claim_once(gate_path, arm, output, release):
    gate = gate_path.resolve(strict=True)
    require(outside_frozen_roots(gate, release), "A run claim must be outside the frozen data roots")
    claim = gate.parent / f"RUN_CLAIM_{sha256_file(gate)}_{arm}.json"
    write_json_exclusive(claim, {"arm": arm, "output": str(output), "authorization_sha256": sha256_file(gate)})


def run(args):
    output = args.output.resolve()
    release = args.release.resolve(strict=True)
    require(outside_frozen_roots(output, release), "Execution may not write into frozen data roots")
    require((args.resume_checkpoint is None) == (args.resume_gate is None) and
            (args.resume_checkpoint is None or args.apply_gate is not None), "Resume requires explicit separate authority")
    protocol = load_protocol(args.protocol)
    validate_identity(args.identity, args.protocol)
    if args.apply_gate:
        authorize_training(args.apply_gate, args.protocol, args.identity, args.arm)
    before = verify_inputs(protocol, release, args.weight, args.deployment_manifest)
    identity = {"protocol_sha256": sha256_file(args.protocol), "identity_sha256": sha256_file(args.identity),
                "deployment_manifest_sha256": sha256_file(args.deployment_manifest), "source_weight_sha256": sha256_file(args.weight)}
    summary = {"status": "PREP_ONLY", "arm": args.arm, **identity, "data_before": before, "training_performed": False,
               "optimizer_updates": 0, "backward_called": False, "val_model_inference_runs": 0, "test_model_inference_runs": 0}
    if not args.apply_gate:
        output.mkdir()
        write_json_exclusive(output / "summary.json", summary)
        return summary

    resume = authorize_resume(args, identity) if args.resume_gate else None
    require(sys.platform.startswith("linux") and output.parent.is_dir() and not output.exists(),
            "Formal execution requires an unused output directory on the pinned Linux runtime")
    require(shutil.disk_usage(output.parent).free >= MIN_RUN_FREE_BYTES,
            "Insufficient free space for retained checkpoints and private formal logs")
    claim_once(args.resume_gate or args.apply_gate, args.arm, output, release)
    output.mkdir()
    write_json_exclusive(output / "process.json", {"host": socket.gethostname(), "pid": os.getpid(),
                         "linux_process_start": linux_process_start(os.getpid())})
    settings = output / "settings"
    os.environ.update(YOLO_CONFIG_DIR=str(settings), MPLCONFIGDIR=str(settings),
                      YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    trainer, error = None, None
    status = HARD_GATE
    start = time.monotonic()
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
        trainer = build_formal_trainer()(release, args.weight.resolve(strict=True), output, args.arm, protocol, identity)
        trainer.run_formal(resume)
        status = PASS_ARM
    except (KeyboardInterrupt, InterruptedError) as failure:
        status, error = INTERRUPTED, f"{type(failure).__name__}: {failure}"
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        (output / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if trainer:
            trainer.event("hard_gate", reason=error)
    finally:
        after = None
        try:
            after = verify_inputs(protocol, release, args.weight, args.deployment_manifest)
            require(after == before, "Runtime input identity changed")
        except Exception as failure:
            status, error = HARD_GATE, f"{error or ''} post-run input verification: {failure}".strip()
        audit = trainer.audit if trainer else None
        counts = dict(trainer.counters) if trainer else {}
        lineage = trainer.previous_segments if trainer else []
        discarded_steps = sum(x["discarded_ledger"]["known_discarded_native_steps"] for x in lineage)
        summary.update(status=status, reason=error, data_after=after, counts=counts,
                       training_performed=counts.get("forwards", 0) > 0, backward_called=counts.get("backwards", 0) > 0,
                       optimizer_attempts=getattr(audit, "optimizer_attempts", 0), native_steps=getattr(audit, "native_steps", 0),
                       optimizer_updates=getattr(audit, "effective_updates", 0), amp_skips=getattr(audit, "amp_skips", 0),
                       optimizer_steps=getattr(audit, "effective_updates", 0),
                       resumed_segments=lineage, known_discarded_native_steps=discarded_steps,
                       known_physical_native_steps=getattr(audit, "native_steps", 0) + discarded_steps,
                       initial_state_sha256=getattr(audit, "initial_state_sha256", None),
                       initial_parameters_sha256=getattr(audit, "initial_parameters_sha256", None),
                       checkpoint=trainer.last_checkpoint if trainer else None,
                       runtime=trainer.runtime if trainer else None,
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if trainer else None,
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if trainer else None,
                       elapsed_seconds=time.monotonic() - start)
        if trainer:
            trainer.close()
        write_json_exclusive(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("protocol", "identity", "release", "weight", "deployment-manifest", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("e0", "e1"))
    for field in ("apply-gate", "resume-checkpoint", "resume-gate"):
        parser.add_argument("--" + field, type=Path)
    summary = run(parser.parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if summary["status"] not in {"PREP_ONLY", PASS_ARM}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
