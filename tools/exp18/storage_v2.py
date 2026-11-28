"""EXP18 V2 storage runner. Only explicitly synthetic CPU execution is enabled."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil

from .common import digest, require, sha256, write_json_exclusive
from .storage_journal import StorageJournal, attempt_id, fsync_directory, lease
from .storage_policy import BATCHES, INTERVAL, MAX_REPLAY_UPDATES, VAL_JOBS, load_storage_profile


def owned_file(root, name):
    require(isinstance(name, str) and name and Path(name).name == name and "/" not in name
            and "\\" not in name and name not in (".", ".."), "Unsafe artifact name")
    path = root / name
    require(path.resolve() == root / name and path.is_file() and not path.is_symlink()
            and path.stat().st_nlink == 1, "Owned artifact missing, linked or escaped")
    return path


def relative_receipt(root, path):
    path = Path(path)
    require(path.parent.resolve() == root, "Receipt path outside its artifact directory")
    path = owned_file(root, path.name)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_receipt(root, receipt):
    path = owned_file(root, receipt["file"])
    require(path.stat().st_size == receipt["bytes"] and sha256(path) == receipt["sha256"],
            "Retained artifact size or SHA changed")
    return path


def retire_payload(root, receipt):
    """Call only after a verified replacement or completed run reserves retirement."""
    path = verify_receipt(root, receipt)
    from .guards import active_guard
    guard = active_guard()
    if guard is not None:
        guard.retire_storage_v2_payload(path, receipt["sha256"])
    else:
        path.unlink()
    fsync_directory(root)


class CheckpointRepository:
    def __init__(self, root, bindings, *, create, max_bytes):
        self.root, self.bindings, self.max_bytes = root, bindings, max_bytes
        if create:
            self.root.mkdir()
            fsync_directory(root.parent)
        require(self.root.resolve() == self.root and self.root.is_dir(), "Unsafe checkpoint repository")

    def _store(self, generation, *, create=False):
        from .checkpoint import ImmutableCheckpointStore
        require(isinstance(generation, str) and len(generation) == 32
                and all(c in "0123456789abcdef" for c in generation), "Invalid owned checkpoint generation")
        folder = self.root / generation
        require(folder.resolve() == folder, "Unsafe checkpoint generation")
        return ImmutableCheckpointStore(folder, self.bindings, create=create, max_bytes=self.max_bytes)

    def publish(self, record):
        generation = attempt_id()
        store = self._store(generation, create=True)
        receipt = store.save(record)
        root = self.root / generation
        fsync_directory(root)
        fsync_directory(self.root)
        return generation, receipt

    def verify(self, spec, *, retired=False):
        store = self._store(spec["generation"])
        root, receipt = store.directory, spec["receipt"]
        commit = owned_file(root, Path(receipt["file"]).with_suffix(".commit.json").name)
        require(json.loads(commit.read_text(encoding="utf-8")) == receipt, "Checkpoint readback receipt changed")
        require(receipt["owner_id"] == store.owner["owner_id"] and receipt["global_update"] == spec["step"]
                and receipt["artifact_kind"] == "synthetic_cpu", "Checkpoint receipt identity changed")
        if retired:
            require(not (root / receipt["file"]).exists() and not (root / receipt["file"]).is_symlink(),
                    "Retired checkpoint payload is unexpectedly present")
        else:
            verify_receipt(root, receipt)

    def scan(self, journal):
        specs = journal.checkpoints
        require({path.name for path in self.root.iterdir()} == {spec["generation"] for spec in specs.values()},
                "Uncommitted or missing checkpoint generation; preserve evidence and HOLD")
        for step, spec in specs.items():
            self.verify(spec, retired=step in journal.retired)

    def load(self, spec):
        self.verify(spec)
        record = self._store(spec["generation"]).load(spec["receipt"])
        require(record["extra"]["journal_prefix"] == spec["prefix"]
                and record["position"]["global_update"] == spec["step"]
                and record["sampler"]["epoch"] == spec["step"] // BATCHES
                and record["sampler"]["next_batch"] == 0, "Checkpoint recovery anchor differs")
        return record

    def retire(self, journal):
        step = min(journal.rolling)
        spec = journal.checkpoints[step]
        sha = spec["receipt"]["sha256"]
        self.verify(journal.checkpoints[journal.persisted])
        self.verify(spec)
        journal.append("RETIRE_BEGIN", step=step, artifact_sha256=sha)
        retire_payload(self.root / spec["generation"], spec["receipt"])
        journal.append("RETIRE_COMMIT", step=step, artifact_sha256=sha)


def verify_validation(root, record):
    wrapper = json.loads(verify_receipt(root, record["artifact"]).read_text(encoding="utf-8"))
    binding = wrapper["binding"]
    require(binding["step"] == record["step"] and binding["epoch"] == record["step"] // BATCHES
            and binding["role"] + "_" + binding["mode"] == record["job"]
            and binding["checkpoint_sha256"] == record["checkpoint_sha256"]
            and binding["evaluator_sha256"] == record["evaluator_sha256"],
            "VAL receipt differs from its planned job")
    artifact = wrapper["evaluation"]["prediction_manifest"]
    expected_dir = f"pred_{binding['epoch']:03d}_{binding['role']}_{binding['mode']}"
    require(artifact["directory"] == expected_dir, "Validation manifest escaped its job")
    directory = root / expected_dir
    require(directory.resolve() == directory, "Unsafe validation directory")
    manifest = json.loads(verify_receipt(directory, {**artifact, "file": "manifest.json"}).read_text(encoding="utf-8"))
    require(manifest["bindings"] == binding and manifest["additional_inference_passes"] == 0, "Validation pass binding differs")
    for image in manifest["images"]:
        require(sha256(owned_file(directory, image["file"])) == image["sha256"], "Saved prediction member changed")
    require(len({r["image_id"] for r in manifest["images"]}) == len(manifest["images"]) > 0,
            "Validation image population is invalid")


def _training_fingerprints(adapter):
    from .bn import cpu_rng_state
    from .checkpoint import state_hash
    bn = {name: value for name, value in adapter.model.state_dict().items()
          if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    return {name: state_hash(value) for name, value in {
        "MASTER": adapter.model.state_dict(), "EMA": adapter.ema.state_dict(), "BN": bn,
        "EMA_model": adapter.ema.ema.state_dict(),
        "AdamW": adapter.optimizer.state_dict(), "Scheduler": adapter.scheduler.state_dict(),
        "AMP": adapter.scaler.state_dict() if adapter.scaler is not None else None,
        "RNG": cpu_rng_state(),
    }.items()}


def _norm(tensors, *, reference):
    import torch
    return (float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(tensors, 2))))
            if tensors else reference)


def _observe(adapter, result):
    import torch
    from collections import defaultdict
    with torch.no_grad():
        domains = defaultdict(list)
        layer16 = []
        for name, parameter in adapter.model.named_parameters():
            parts = name.split(".")
            if parts[0] == "model" and parts[1].isdigit():
                index = int(parts[1])
                domain = "backbone" if index <= 10 else "neck" if index <= 16 else "detection"
                if index == 16:
                    layer16.append(parameter.detach())
            else:
                domain = parts[0] if parts[0] in ("backbone", "neck", "region_projection", "state_head") else "other"
            if parameter.grad is not None:
                domains[domain].append(parameter.grad.detach())
        gap = getattr(result, "gap", None)
        classes = getattr(result, "pair_classes", ())
        if gap is not None:
            gap = gap.detach().reshape(-1)
        polygons, summaries = Counter(str(value) for value in classes), {}
        if classes:
            require(gap is not None and len(gap) == len(classes), "A1 diagnostic class and gap count differ")
            losses = result.pair_loss.detach().reshape(-1).cpu().tolist()
            margins = gap.cpu().tolist()
            for cls, count in polygons.items():
                members = [i for i, value in enumerate(classes) if str(value) == cls]
                summaries[cls] = {"count": count, "loss_sum": sum(losses[i] for i in members),
                                  "gap_sum": sum(margins[i] for i in members),
                                  "positive": sum(margins[i] > 0 for i in members)}
        return {"auxiliary_polygons_by_class": dict(polygons), "auxiliary_polygons": int(gap.numel()) if gap is not None else 0,
                "score_gap_mean": float(gap.mean()) if gap is not None else None,
                "pair_order_accuracy": float((gap > 0).float().mean()) if gap is not None else None,
                "pair_by_class": summaries,
                "joint_gradient_domains_preclip": {k: _norm(v, reference=0.) for k, v in sorted(domains.items())},
                "neck16_parameter_norm": _norm(layer16, reference=None)}


def _summary(rows):
    from statistics import median
    result = {}
    for field in ("yolo_loss", "supcon_loss", "state_loss", "gradient_norm", "clip_coefficient", "score_gap_mean",
                  "pair_order_accuracy", "neck16_parameter_norm"):
        values = [row[field] for row in rows if row.get(field) is not None]
        result[field] = ({"count": len(values), "min": min(values), "median": median(values), "max": max(values)}
                         if values else {"count": 0})
    classes = Counter()
    for row in rows:
        classes.update(row["auxiliary_polygons_by_class"])
    pair_by_class, domain_norms = {}, {}
    for row in rows:
        for cls, value in row["pair_by_class"].items():
            pair_by_class.setdefault(cls, Counter()).update(value)
        for domain, norm in row["joint_gradient_domains_preclip"].items():
            domain_norms.setdefault(domain, []).append(norm)
    return {"distributions": result, "steps": len(rows), "class_exposures": dict(classes),
            "pair_by_class": pair_by_class,
            "joint_gradient_median_preclip": {k: median(v) for k, v in domain_norms.items()},
            "BN_start_sha256": next(iter(rows))["state"]["BN"],
            "BN_end_sha256": list(rows)[-1]["state"]["BN"],
            "amp_skips": sum(row["amp_skips"] for row in rows),
            "optimizer_component_gradients": "NOT_MEASURED_NO_EXTRA_BACKWARD"}


class StorageExecutor:
    """Drive the existing joint-update core with the V2 checkpoint and VAL cadence."""

    def __init__(self, adapter, protocol, bindings, root, journal, repository, *, failpoint=None):
        from .execution import ExperimentExecutor
        self.adapter = adapter
        self.root, self.bindings, self.journal, self.repository = root, bindings, journal, repository
        self.core = ExperimentExecutor(adapter, protocol, bindings, None, journal, synthetic=True)
        self.failpoint = failpoint or (lambda _event, _step: None)

    def _artifact(self, name, payload):
        path = self.root / "artifacts" / name
        write_json_exclusive(path, payload)
        fsync_directory(path.parent)
        return relative_receipt(path.parent, path)

    def _checkpoint(self):
        journal, step = self.journal, self.core.global_update
        epoch, prefix = step // BATCHES, journal.prefix
        record = self.adapter.capture(position={"epoch": epoch, "batch_position": 0, "global_update": step},
            sampler_state={"epoch": epoch, "next_batch": 0, "contract": self.bindings["data_sha256"]},
            head_active_updates=self.core.head_active_updates, extra={"journal_prefix": prefix})
        self.failpoint("before_checkpoint", step)
        generation, receipt = self.repository.publish(record)
        self.failpoint("after_checkpoint_blob", step)
        kind = "permanent" if epoch in journal.policy["permanent_epochs"] else "rolling"
        from .checkpoint import state_hash
        role_sha = {"MASTER": state_hash(record["state"]), "EMA": state_hash(record["ema"]["state"])}
        journal.append("CHECKPOINT_COMMIT", step=step, kind=kind, attempt=journal.run_attempt,
                       generation=generation, receipt=receipt, prefix=prefix, role_state_sha256=role_sha)
        self.failpoint("after_checkpoint_commit", step)
        if len(journal.rolling) > 2:
            self.repository.retire(journal)

    def _restore(self):
        journal = self.journal
        step = journal.persisted
        require(journal.pending_val is None and journal.pending_retire is None and journal.pending_recovery is None,
                "Interrupted VAL, retirement or restore; preserve evidence and HOLD")
        require(journal.executed - step <= MAX_REPLAY_UPDATES, "Recovery would exceed its operation budget")
        if journal.executed > step or journal.pending_step is not None:
            journal.append("SEGMENT_ABANDONED", step=journal.executed, anchor=step, attempt=journal.run_attempt,
                           pending_step=journal.pending_step)
        new_attempt = attempt_id()
        journal.append("RECOVERY_BEGIN", step=step, attempt=new_attempt, previous_attempt=journal.run_attempt)
        record = self.repository.load(journal.checkpoints[step])
        state = self.adapter.restore(record)
        self.core.global_update = state["position"]["global_update"]
        self.core.head_active_updates = state["head_active_updates"]
        require(self.core.global_update == step, "Restored logical update disagrees")
        journal.append("RECOVERY_COMPLETE", step=step, attempt=new_attempt,
                       checkpoint_sha256=journal.checkpoints[step]["receipt"]["sha256"])

    def _evaluate_boundary(self):
        journal, step = self.journal, self.journal.persisted
        if step == 0 or journal.boundary_evaluation_complete:
            return
        epoch = step // BATCHES
        checkpoint = journal.checkpoints[step]["receipt"]
        if step not in journal.diagnostics:
            interval = {}
            for event in journal.rows:
                if event["event"] == "UPDATE_EXECUTED":
                    pos = event["fields"]["step"]
                    if step - BATCHES * INTERVAL < pos <= step:
                        interval[pos] = event["fields"]["evidence"]
            require(len(interval) == BATCHES * INTERVAL, "Diagnostic interval is incomplete")
            artifact = self._artifact(f"diagnostic_{epoch:03d}.json", _summary(interval.values()))
            journal.append("DIAGNOSTIC_COMMIT", step=step, checkpoint_sha256=checkpoint["sha256"], artifact=artifact)
        for job in VAL_JOBS:
            if (step, job) in journal.val:
                continue
            role, mode = job.split("_")
            snapshot = _training_fingerprints(self.adapter)
            role_sha = snapshot["EMA_model"] if role == "EMA" else snapshot["MASTER"]
            require(role_sha == journal.checkpoints[step]["role_state_sha256"][role],
                    "VAL source differs from its saved Checkpoint")
            protected = {"checkpoint_sha256": checkpoint["sha256"], "evaluation_state_sha256": role_sha,
                         "evaluator_sha256": self.bindings["evaluator_sha256"],
                         "epoch": epoch, "step": step, "role": role, "mode": mode}
            journal.append("VALIDATION_BEGIN", step=step, job=job, checkpoint_sha256=checkpoint["sha256"])
            self.failpoint("after_validation_begin", step)
            value = self.adapter.evaluate_v2(epoch, role, mode, protected)
            require(_training_fingerprints(self.adapter) == snapshot, "Evaluation modified protected training state")
            artifact = self._artifact(f"validation_{epoch:03d}_{job}.json", {"binding": protected, "evaluation": value})
            fields = {"step": step, "job": job, "checkpoint_sha256": checkpoint["sha256"],
                      "evaluator_sha256": self.bindings["evaluator_sha256"], "artifact": artifact}
            verify_validation(self.root / "artifacts", fields)
            journal.append("VALIDATION_COMMIT", **fields)
            self.failpoint("after_validation_commit", step)
        if step not in journal.best:
            selections = self.adapter.observe_best_v2(epoch, checkpoint, {
                job: journal.val[step, job]["artifact"] for job in VAL_JOBS
            })
            for receipt in selections.values():
                verify_receipt(self.root / "artifacts", receipt)
            journal.append("BEST_COMMIT", step=step, selections=selections)

    def _step(self, cursor, epoch):
        journal = self.journal
        step = epoch * BATCHES + cursor.next_index + 1
        packet = cursor.next_batch()
        metadata = self.adapter.packet_evidence_v2(packet)
        require(set(metadata) >= {"images_sha256", "augmentation_sha256", "A_sha256", "N_sha256", "matching_sha256"},
                "Missing frozen batch identity")
        journal.append("ATTEMPT_BEGIN", step=step, attempt=journal.run_attempt)
        self.failpoint("after_attempt_begin", step)
        row = self.core._update(packet, step, observe=lambda result: _observe(self.adapter, result))
        require(row["matching_sha256"] == metadata["matching_sha256"], "Forward used a different matching map")
        require(set(row).isdisjoint(metadata.keys() - {"matching_sha256"}), "Batch evidence overrides update metrics")
        evidence = {**row, **metadata, "epoch": epoch + 1, "batch_position": (step - 1) % BATCHES + 1,
                    "arm_id": self.bindings["arm_id"],
                    "lr": [group["lr"] for group in self.adapter.optimizer.param_groups],
                    "state": _training_fingerprints(self.adapter), "amp_skips": 0,
                    "component_gradient_norms": None, "checkpoint_anchor_sha256": journal.checkpoints[journal.persisted]["receipt"]["sha256"],
                    "resource": self.adapter.resources_v2()}
        replay = {k: v for k, v in evidence.items() if k not in ("resource", "checkpoint_anchor_sha256")}
        journal.append("UPDATE_EXECUTED", step=step, attempt=journal.run_attempt,
                       evidence=evidence, replay_sha256=digest(replay))
        self.failpoint("after_update", step)

    def run(self, *, resume):
        journal = self.journal
        require(not journal.complete, "The completed run is immutable")
        self.repository.scan(journal)
        self.adapter.verify_storage_v2(self.root, journal)
        for fields in journal.val.values():
            verify_validation(self.root / "artifacts", fields)
        for selections in journal.best.values():
            for receipt in selections.values():
                verify_receipt(self.root / "artifacts", receipt)
        for receipt in journal.diagnostics.values():
            verify_receipt(self.root / "artifacts", receipt)
        if resume:
            require(journal.checkpoints, "No committed initial state")
            self._restore()
            if len(journal.rolling) > 2:
                self.repository.retire(journal)
            self._evaluate_boundary()
        else:
            journal.append("RUN_BEGIN", attempt=attempt_id())
            self._checkpoint()
        for epoch in range(journal.persisted // BATCHES, self.core.epochs):
            self.adapter.start_epoch(epoch)
            cursor = self.adapter.build_cache(epoch)
            while cursor.next_index < BATCHES:
                self._step(cursor, epoch)
            if (epoch + 1) % INTERVAL == 0:
                self._checkpoint()
                self._evaluate_boundary()
        endpoints = self.adapter.export_last_v2(self.root, self.repository, journal.checkpoints[journal.final_update])
        for receipt in endpoints.values():
            verify_receipt(self.root / "artifacts", receipt)
        self.repository.verify(journal.checkpoints[journal.final_update])
        journal.append("TRAIN_COMPLETE", step=journal.final_update, endpoints=endpoints)
        while journal.rolling:
            self.repository.retire(journal)
        return {"status": "CPU_SYNTHETIC_COMPLETE", "updates": journal.persisted, "validation_jobs_mocked": len(journal.val),
                "permanent_checkpoints": sorted(set(journal.checkpoints) - journal.retired)}


def run_storage_cpu(adapter_factory, *, profile, repository, output, bindings, resume=False, failpoint=None,
                    checkpoint_max_bytes=16 * 1024**2, minimum_free_bytes=64 * 1024**2):
    runtime, protocol = load_storage_profile(profile, repository)
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_initialized(), "V2 requires a CPU-only Torch build")
    require(not Path(output).is_symlink(), "Storage run root is a symlink")
    root = Path(output).resolve()
    owned = Path(repository).resolve() / "private_assets" / "exp18_storage_v2"
    require(root.is_relative_to(owned) and root != owned and not owned.is_symlink(), "Storage run escaped its private root")
    require(bindings.get("arm_id") in protocol["arms"], "V2 execution needs a frozen experiment arm")
    bindings = {**copy.deepcopy(bindings), "execution_id": root.name, "storage_sha256": runtime["storage_sha256"],
                "protocol_sha256": runtime["base_protocol_sha256"]}
    require(all(isinstance(bindings.get(k), str) and len(bindings[k]) == 64
                for k in ("code_sha256", "data_sha256", "matching_sha256", "evaluator_sha256")),
            "Incomplete execution identity")
    if not resume:
        root.mkdir(parents=True)
        fsync_directory(root.parent)
    require(shutil.disk_usage(root).free >= minimum_free_bytes, "Storage capacity reserve reached")
    with lease(root, create=not resume):
        owner = {"bindings": bindings, "artifact_kind": "synthetic_cpu_v2"}
        if not resume:
            write_json_exclusive(root / "owner.json", owner)
            (root / "artifacts").mkdir()
            fsync_directory(root)
        require(json.loads(owned_file(root, "owner.json").read_text(encoding="utf-8")) == owner,
                "V2 run identity differs")
        journal = StorageJournal(root / "journal.jsonl", bindings, runtime["storage"], create=not resume)
        store = CheckpointRepository(root / "checkpoints", bindings, create=not resume, max_bytes=checkpoint_max_bytes)
        adapter = adapter_factory(bindings)
        require(getattr(adapter.model, "exp18_synthetic", False) is True
                and all(p.device.type == "cpu" for p in adapter.model.parameters()), "V2 runner accepts synthetic CPU only")
        require(not any(root.glob("hold_*.json")), "Execution has an existing HOLD receipt")
        try:
            return StorageExecutor(adapter, protocol, bindings, root, journal, store, failpoint=failpoint).run(resume=resume)
        except (ValueError, OSError, RuntimeError) as exc:
            write_json_exclusive(root / f"hold_{attempt_id()}.json", {
                "status": "HOLD", "reason": str(exc), "exception": type(exc).__name__,
                "journal_prefix": journal.prefix, "persisted_update": journal.persisted,
                "executed_update": journal.executed, "pending_update": journal.pending_step,
            })
            raise
