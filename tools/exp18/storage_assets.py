"""V2 finite retention for owned epoch packets and supplementary Best weights."""
from collections import Counter
import json
import math
import os
from pathlib import Path

from .common import digest, require, write_json_exclusive
from .storage_journal import attempt_id, fsync_directory
from .storage_policy import BATCHES
from .storage_v2 import owned_file, relative_receipt, retire_payload, verify_receipt


def save_cpu_weights(path, payload):
    import torch
    from .checkpoint import state_hash
    from .guards import active_guard
    from .common import sha256
    require(all(tensor.device.type == "cpu" and (not tensor.is_floating_point() or
                tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()))
                for tensor in payload["state"].values()), "Export requires finite CPU FP32 weights")
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    guard = active_guard()
    if guard is not None:
        guard.allow_cpu_load(path, sha256(path), kind="synthetic_CPU_checkpoint")
    require(state_hash(torch.load(path, map_location="cpu", weights_only=True)) == state_hash(payload),
            "Model-only artifact failed readback")
    fsync_directory(path.parent)
    return relative_receipt(path.parent, path)


def make_last(root, role, state, metadata):
    require(role in ("MASTER", "EMA"), "Invalid Last role")
    path = root / f"last_{role.lower()}.pt"
    return save_cpu_weights(path, {"schema": "EXP18_V2_LAST", "state": state, "metadata": metadata, "role": role})


def _read(root, name):
    return json.loads(owned_file(root, name).read_text(encoding="utf-8"))


class BestWeights:
    """The active logical package has two independently selected role entries."""

    def __init__(self, root):
        self.root = root

    def _selections(self):
        choices = {}
        version = 0
        records = sorted(self.root.glob("best_selection_*.json"))
        expected_weights = set()
        for path in records:
            record = _read(self.root, path.name)
            version += 1
            role = record["role"]
            require(path.name == f"best_selection_{version:04d}.json" and role in ("MASTER", "EMA"),
                    "Best selection sequence differs")
            prior = choices.get(role)
            require(record["prior"] == (prior["record"] if prior else None), "Best selection previous identity differs")
            if record["weights"] is not None:
                expected_weights.add(record["weights"]["file"])
            choices[role] = {**record, "record": path.name}
        for record in choices.values():
            if record["weights"] is not None:
                verify_receipt(self.root, record["weights"])
        existing_weights = {p.name for p in self.root.glob("best_weights_*.pt")}
        active_weights = {record["weights"]["file"] for record in choices.values() if record["weights"] is not None}
        require(existing_weights == active_weights and existing_weights <= expected_weights,
                "Interrupted Best publication or retirement; HOLD")
        return choices, version

    def observe(self, *, epoch, role, metrics, state, checkpoint, permanent, evaluation, metadata=None):
        require(role in ("MASTER", "EMA"), "Invalid Best role")
        choices, version = self._selections()
        values = [metrics[k] for k in ("mask_map", "box_map")]
        require(all(type(value) in (int, float) and math.isfinite(value) for value in values),
                "Best fitness is nonfinite")
        fitness = sum(values)
        prior = choices.get(role)
        if prior is not None:
            require(epoch >= prior["epoch"], "Best epoch moved backward")
            if epoch == prior["epoch"]:
                require(evaluation == prior["evaluation"], "Best retry evaluation differs")
                return prior
            if fitness < prior["fitness"]:
                return prior
        weights = None
        if not permanent:
            weights = save_cpu_weights(self.root / f"best_weights_{version + 1:04d}_{role.lower()}.pt",
                {"schema": "EXP18_V2_BEST", "state": state, "epoch": epoch, "role": role, "metadata": metadata})
        path = self.root / f"best_selection_{version + 1:04d}.json"
        record = {"epoch": epoch, "role": role, "fitness": fitness, "metrics": metrics,
                  "prior": prior["record"] if prior else None, "weights": weights,
                  "checkpoint": checkpoint, "permanent_reference": permanent, "evaluation": evaluation}
        write_json_exclusive(path, record)
        fsync_directory(self.root)
        if prior is not None and prior["weights"] is not None:
            retire_payload(self.root, prior["weights"])
        return {**record, "record": path.name}

    def active(self):
        return self._selections()[0]


class EpochPacketCache:
    """Retain two sealed epochs and all manifests. Rebuilds get a new generation."""

    def __init__(self, root, bindings, *, maximum_epoch_bytes):
        self.root, self.bindings, self.maximum_epoch_bytes = root, digest(bindings), maximum_epoch_bytes
        require(root.resolve() == root and root.is_dir() and not root.is_symlink(), "Unsafe cache root")
        self._live_catalog = None

    def _catalog(self):
        commits = self.root / "_commits"
        folders = sorted(p for p in self.root.iterdir() if p != commits)
        if not commits.exists():
            require(not folders, "Epoch cache is missing generation receipts")
            commits.mkdir()
            fsync_directory(self.root)
        require(commits.resolve() == commits and commits.is_dir(), "Unsafe epoch cache commit directory")
        generations = []
        require({p.name for p in commits.iterdir()} == {f"{folder.name}.json" for folder in folders},
                "Uncommitted or missing epoch cache generation")
        for index, folder in enumerate(folders):
            require(folder.is_dir() and not folder.is_symlink() and folder.name.startswith(f"{index:06d}_"),
                    "Invalid epoch generation")
            manifest = _read(folder, "manifest.json")
            require(manifest["bindings"] == self.bindings and len(manifest["packets"]) == BATCHES
                    and _read(commits, f"{folder.name}.json") == {"manifest_sha256": digest(manifest)},
                    "Cache manifest identity differs")
            for i, receipt in enumerate(manifest["packets"]):
                require(receipt["file"] == f"batch_{i:03d}.packet", "Unexpected cache member")
            retired = (folder / "retire_commit.json").exists()
            require(retired == (folder / "retire_begin.json").exists(), "Interrupted cache retirement")
            if retired:
                ticket = _read(folder, "retire_commit.json")
                require(ticket == _read(folder, "retire_begin.json") == {"manifest_sha256": digest(manifest)}
                        and all(not (folder / row["file"]).exists() for row in manifest["packets"]),
                        "Retired epoch package is inconsistent")
            generations.append((folder, manifest, retired))
        live = [row for row in generations if not row[2]]
        require(len(live) <= 2, "Unfinished epoch cache rotation")
        return generations, live

    def open_or_build(self, epoch, expected, prepare, *, fingerprint):
        require(type(epoch) is int and epoch >= 0, "Invalid cache epoch")
        require(len(expected) == BATCHES, "Cache requires an accepted 21-batch content ledger")
        if self._live_catalog is None:
            self._live_catalog = self._catalog()
            for folder, manifest, _ in self._live_catalog[1]:
                for row in manifest["packets"]:
                    verify_receipt(folder, row)
        generations, live = self._live_catalog
        for folder, manifest, _ in reversed(live):
            if manifest["epoch"] == epoch:
                require([row["content_sha256"] for row in manifest["packets"]] == list(expected),
                        "Cached epoch differs from the sealed content ledger")
                for row in manifest["packets"]:
                    verify_receipt(folder, row)
                return folder, manifest
        folder = self.root / f"{len(generations):06d}_{attempt_id()}"
        folder.mkdir()
        fsync_directory(self.root)
        rows, size = [], 0
        for i, (wire, content_sha) in enumerate(prepare()):
            require(i < BATCHES and content_sha == expected[i] and fingerprint(wire) == content_sha,
                    "Rebuilt batch content SHA differs")
            size += len(wire)
            require(size <= self.maximum_epoch_bytes, "Prepared epoch exceeds measured byte budget")
            target = folder / f"batch_{i:03d}.packet"
            with target.open("xb") as stream:
                stream.write(wire)
                stream.flush()
                os.fsync(stream.fileno())
            rows.append({**relative_receipt(folder, target), "content_sha256": content_sha})
        require(len(rows) == BATCHES, "Incomplete prepared epoch")
        manifest = {"bindings": self.bindings, "epoch": epoch, "packets": rows}
        write_json_exclusive(folder / "manifest.json", manifest)
        fsync_directory(folder)
        commits = self.root / "_commits"
        write_json_exclusive(commits / f"{folder.name}.json", {"manifest_sha256": digest(manifest)})
        fsync_directory(commits)
        if len(live) == 2:
            older, old, _ = live[0]
            for row in old["packets"]:
                verify_receipt(older, row)
            ticket = {"manifest_sha256": digest(old)}
            write_json_exclusive(older / "retire_begin.json", ticket)
            fsync_directory(older)
            for row in old["packets"]:
                retire_payload(older, row)
            write_json_exclusive(older / "retire_commit.json", ticket)
            fsync_directory(older)
            i = next(i for i, row in enumerate(generations) if row[0] == older)
            generations[i] = (older, old, True)
            live.pop(0)
        new_row = (folder, manifest, False)
        generations.append(new_row)
        live.append(new_row)
        return folder, manifest
