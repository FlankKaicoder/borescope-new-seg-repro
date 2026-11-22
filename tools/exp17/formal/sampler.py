"""Seal the unchanged native worker stream once; recover from exact CPU bytes.

Native J1 uses two workers, so a shuffle cursor alone cannot restore prefetched
worker RNG. The formal adapter drains that *same* loader into an immutable
epoch cache before any optimizer update for that epoch. Consumption and
recovery use its sealed bytes, never replay completed transforms or raw reads.
This changes loader/optimizer overlap, not sampling or augmentation semantics.
"""
import copy
import dataclasses
import json
import os
from pathlib import Path

import torch

from tools.exp17.formal.checkpoint import atomic_json, fsync_directory, require, sha256_file, state_hash
from tools.exp17.formal.rng import capture_rng, rng_equal

SCHEMA = "EXP17_G1_SEALED_NATIVE_EPOCH_CACHE_V1"
CURSOR_SCHEMA = "EXP17_G1_SEALED_BATCH_CURSOR_V1"


def _assert_cpu(value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "Epoch cache contains a non-CPU tensor")
    elif isinstance(value, dict):
        for child in value.values(): _assert_cpu(child)
    elif isinstance(value, (list, tuple)):
        for child in value: _assert_cpu(child)
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value): _assert_cpu(getattr(value, field.name))


class SealedEpochCache:
    @classmethod
    def build(cls, directory, loader, *, epoch, bindings, batch_fingerprint, expected_batches=21):
        require(isinstance(epoch, int) and epoch >= 0 and expected_batches > 0, "Invalid epoch cache specification")
        directory = Path(directory)
        require(not directory.is_symlink(), "Epoch cache directory is a symlink")
        directory.mkdir(parents=True, exist_ok=False)
        before = capture_rng()
        rows = []
        for number, batch in enumerate(loader):
            require(number < expected_batches, "Native epoch loader delivered too many batches")
            _assert_cpu(batch)
            # Caller verifies the full frozen native ledger BEFORE sealing.
            fingerprint = batch_fingerprint(number, batch)
            require(isinstance(fingerprint, (dict, str)), "Missing caller-verified native batch fingerprint")
            path = directory / f"batch_{number:03d}.pt"
            with path.open("xb") as stream:
                torch.save(batch, stream); stream.flush(); os.fsync(stream.fileno())
            rows.append({"index": number, "file": path.name, "sha256": sha256_file(path),
                         "bytes": path.stat().st_size, "native_fingerprint": copy.deepcopy(fingerprint)})
        require(len(rows) == expected_batches, "Native epoch loader delivered incomplete batches")
        require(rng_equal(before, capture_rng()), "Native worker stream cache consumed main training RNG")
        manifest = {"schema": SCHEMA, "epoch": epoch, "bindings": copy.deepcopy(bindings),
                    "batches": rows, "expected_batches": expected_batches,
                    "main_RNG_unchanged": True, "native_loader_unmodified": True,
                    "cache_pickle_policy": "ONLY_AFTER_EXACT_SEALED_SHA_BINDING; trusted local ReplayTrace objects"}
        atomic_json(directory / "manifest.json", manifest)
        return cls.open(directory, bindings=bindings, expected_manifest_sha256=sha256_file(directory / "manifest.json"))

    @classmethod
    def open(cls, directory, *, bindings, expected_manifest_sha256):
        directory = Path(directory)
        require(not directory.is_symlink() and directory.is_dir(), "Epoch cache path changed")
        path = directory / "manifest.json"
        require(path.is_file() and not path.is_symlink() and sha256_file(path) == expected_manifest_sha256, "Epoch cache manifest hash conflict")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        require(manifest.get("schema") == SCHEMA and manifest.get("bindings") == bindings, "Epoch cache code/data/protocol binding mismatch")
        rows = manifest["batches"]
        require(len(rows) == manifest["expected_batches"] and [r["index"] for r in rows] == list(range(len(rows))), "Epoch cache has missing or reordered batches")
        require(all(r["file"] == f"batch_{n:03d}.pt" for n, r in enumerate(rows)), "Unsafe epoch cache member path")
        cache = cls()
        cache.directory, cache.manifest, cache.manifest_sha256 = directory.resolve(), manifest, expected_manifest_sha256
        return cache

    def cursor(self, next_batch=0):
        return CachedBatchCursor(self, next_batch=next_batch)

    def retire(self, *, owned_cache_root, committed_sampler_state, retention_authorized=False):
        """Future-Gate-only retirement of exact generated cache members.

        This never runs during G1. The latest committed checkpoint must refer
        to a cache at least two epochs newer; it retains current+previous. It
        refuses unexpected members and never recursively traverses a path.
        """
        require(retention_authorized is True, "Active-cache retirement was not explicitly authorized")
        root = Path(owned_cache_root).resolve(strict=True)
        require(not self.directory.is_symlink() and self.directory.resolve().parent == root
                and self.directory.name == f"epoch_{self.manifest['epoch']:03d}", "Cache retirement escaped its owned run root")
        require(committed_sampler_state["schema"] == CURSOR_SCHEMA
                and committed_sampler_state["epoch"] >= self.manifest["epoch"] + 2
                and committed_sampler_state["cache_manifest_sha256"] != self.manifest_sha256, "A committed checkpoint still needs this cache")
        require(sha256_file(self.directory / "manifest.json") == self.manifest_sha256, "Cache changed before retirement")
        allowed = {"manifest.json", *(r["file"] for r in self.manifest["batches"])}
        require({p.name for p in self.directory.iterdir()} == allowed, "Unexpected cache assets; preserve directory")
        for row in self.manifest["batches"]:
            path = self.directory / row["file"]
            require(not path.is_symlink() and path.is_file() and sha256_file(path) == row["sha256"], "Cache member changed before retirement")
        # Preflight every exact member before the first operation.
        for name in sorted(allowed): (self.directory / name).unlink()
        self.directory.rmdir(); fsync_directory(root)


class CachedBatchCursor:
    def __init__(self, cache, *, next_batch=0):
        self.cache = cache
        require(isinstance(next_batch, int) and 0 <= next_batch <= len(cache.manifest["batches"]), "Invalid cache batch cursor")
        self.next_index = next_batch

    def state_dict(self):
        return {"schema": CURSOR_SCHEMA, "epoch": self.cache.manifest["epoch"],
                "cache_directory": str(self.cache.directory), "cache_manifest_sha256": self.cache.manifest_sha256,
                "next_batch": self.next_index, "num_batches": len(self.cache.manifest["batches"]),
                "last_batch_sha256": self.cache.manifest["batches"][self.next_index - 1]["sha256"] if self.next_index else None}

    def load_state_dict(self, state):
        require(state["schema"] == CURSOR_SCHEMA and state["epoch"] == self.cache.manifest["epoch"]
                and Path(state["cache_directory"]).resolve() == self.cache.directory
                and state["cache_manifest_sha256"] == self.cache.manifest_sha256
                and state["num_batches"] == len(self.cache.manifest["batches"]), "Sampler/cache binding changed at recovery")
        previous = self.next_index
        candidate = state["next_batch"]
        require(isinstance(candidate, int) and not isinstance(candidate,bool)
                and 0 <= candidate <= len(self.cache.manifest["batches"]), "Invalid recovered sampler cursor")
        self.next_index = candidate
        if self.state_dict() != state:
            self.next_index = previous
            raise ValueError("Sampler cursor or prior batch identity changed")

    def next_batch(self):
        rows = self.cache.manifest["batches"]
        if self.next_index == len(rows): raise StopIteration
        require(sha256_file(self.cache.directory / "manifest.json") == self.cache.manifest_sha256, "Epoch cache manifest changed while consuming")
        row = rows[self.next_index]; path = self.cache.directory / row["file"]
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == row["bytes"]
                and sha256_file(path) == row["sha256"], "Augmented batch cache corruption/hash conflict")
        # Opaque ReplayTrace dataclasses are allowed ONLY after byte identity
        # from this same run's sealed, protocol/code-bound private manifest.
        batch = torch.load(path, map_location="cpu", weights_only=False)
        _assert_cpu(batch)
        self.next_index += 1
        return batch

    def __iter__(self): return self
    def __next__(self): return self.next_batch()
