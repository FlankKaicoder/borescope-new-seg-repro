"""V2 transaction/recovery acceptance with synthetic CPU tensors and callbacks."""
from collections import Counter
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import warnings

import numpy as np
import torch
from torch import nn

from tools.exp18.checkpoint import capture_training_state, restore_training_state, state_hash
from tools.exp18.common import digest, sha256, write_json_exclusive
from tools.exp18.guards import CPUExecutionGuard
from tools.exp18.model import add_state_head, native_state_dict
from tools.exp18.optimizer import build_adamw, CPUModelEMA, HeadAwareEMA
from tools.exp18.storage_assets import BestWeights, EpochPacketCache, make_last
from tools.exp18.storage_journal import lease, StorageJournal
from tools.exp18.storage_policy import load_storage_profile, policy
from tools.exp18.storage_v2 import run_storage_cpu, _training_fingerprints, relative_receipt, verify_receipt
from tools.exp18.validation import PredictionWriter

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / "private_assets/exp18_storage_v2"
PROFILE = REPO / "configs/exp18/runtime_storage_v2_h200.json"
BINDINGS = {"code_sha256": "a"*64, "data_sha256": "b"*64, "matching_sha256": "c"*64,
            "evaluator_sha256": sha256(__file__), "arm_id": "a1_true"}


class InjectedCrash(Exception):
    pass


class Toy(nn.Module):
    exp18_synthetic = True

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 4)
        self.bn = nn.BatchNorm1d(4)
        self.neck = nn.Linear(4, 64)
        self.region_projection = nn.Linear(64, 3)
        add_state_head(self, seed=42)


class Cursor:
    def __init__(self, folder, manifest):
        self.folder, self.manifest, self.next_index = folder, manifest, 0

    def next_batch(self):
        row = self.manifest["packets"][self.next_index]
        wire = verify_receipt(self.folder, row).read_bytes()
        value = json.loads(wire)
        if digest(value) != row["content_sha256"]:
            raise ValueError("Synthetic packet content changed")
        self.next_index += 1
        return value


class SyntheticAdapter:
    def __init__(self, root, bindings, *, val_calls, crash=None):
        self.root, self.bindings, self.val_calls, self.crash = root, bindings, val_calls, crash
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        self.model = Toy().train()
        self.optimizer, _ = build_adamw(self.model)
        self.ema = HeadAwareEMA(CPUModelEMA(self.model))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda e: 1 - .002 * e)
        self.scaler = torch.amp.GradScaler("cpu", init_scale=1., growth_interval=2000)
        self.counters = {"optimizer_attempts": 0, "amp_skips": 0}
        self.forward_calls = 0

    def verify_storage_v2(self, root, _journal):
        cache = root / "cache"
        cache.mkdir(exist_ok=True)
        self.cache = EpochPacketCache(cache, self.bindings, maximum_epoch_bytes=1024**2)
        self.best = BestWeights(root / "artifacts")
        self.best.active()

    def start_epoch(self, _epoch):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of")
            self.scheduler.step()

    def build_cache(self, epoch):
        packets = [{"step": epoch * 21 + i + 1, "epoch": epoch, "i": i} for i in range(21)]
        rows = [digest(row) for row in packets]
        folder, manifest = self.cache.open_or_build(epoch, rows,
            lambda: ((json.dumps(packet).encode(), sha) for packet, sha in zip(packets, rows, strict=True)),
            fingerprint=lambda wire: digest(json.loads(wire)))
        return Cursor(folder, manifest)

    def packet_evidence_v2(self, packet):
        return {k + "_sha256": digest([k, packet["step"]]) for k in ("images", "augmentation", "A", "N", "matching")}

    def before_batch(self, step):
        if step != self.counters["optimizer_attempts"] + 1:
            raise ValueError("Optimizer step repeated or skipped")

    def forward_batch(self, packet, step):
        self.forward_calls += 1
        if step != packet["step"]:
            raise ValueError("Synthetic input order changed")
        active = step % 3 != 2
        x = torch.randn(4, 2) + float(np.random.random()) + random.random()
        feature = self.model.neck(self.model.bn(self.model.backbone(x)))
        yolo = feature.square().mean()
        supcon = self.model.region_projection(feature).square().mean()
        state = yolo.new_zeros(())
        gap = None
        if active:
            gap = self.model.state_head(feature[0:2]) - self.model.state_head(feature[2:4])
            state = torch.nn.functional.softplus(-gap).mean()
        total = yolo + 8 * supcon + .5 * state
        if self.crash == "NONFINITE":
            total = total * float("inf")
        return SimpleNamespace(head_active=active, yolo=yolo, supcon=supcon, state=state, total=total,
                               gap=gap, pair_loss=torch.nn.functional.softplus(-gap) if active else None,
                               pair_classes=(0, 0) if active else (), matching_sha256=digest(["matching", step]))

    def after_batch(self, row):
        self.counters["optimizer_attempts"] = row["global_update"]

    def resources_v2(self):
        return {"type": "SYNTHETIC_CPU", "gpu_allocated": 0}

    def capture(self, *, position, sampler_state, head_active_updates, extra):
        return capture_training_state(self.model, self.optimizer, self.scheduler, self.ema,
            bindings=self.bindings, position=position, sampler_state=sampler_state, scaler=self.scaler,
            head_active_updates=head_active_updates, counters=self.counters, extra=extra)

    def restore(self, record):
        state = restore_training_state(self.model, self.optimizer, self.scheduler, self.ema, record,
                                      bindings=self.bindings, scaler=self.scaler)
        self.counters = state["counters"]
        return state

    def evaluate_v2(self, epoch, role, mode, bindings):
        self.val_calls.append((epoch, role, mode))
        if self.crash == "VAL_MUTATION":
            self.model.bn.running_mean.add_(.1)
        # Exercise the same streaming lossless prediction writer with synthetic images.
        directory = self.root / "artifacts" / f"pred_{epoch:03d}_{role}_{mode}"
        writer = PredictionWriter(directory, bindings, synthetic=True)
        masks = np.ones((1, 16, 16), dtype=np.uint8)
        prediction = {"cls": np.array([0]), "conf": np.array([.8]),
                      "bboxes": np.array([[0, 0, 16, 16]]), "masks": masks}
        target = {key: prediction[key] for key in ("cls", "bboxes", "masks")}
        writer.add("synthetic-image", prediction, target, gt_ordinals=[0], thresholds=[.5, .75],
                   mask_iou=np.ones((1, 1)), box_iou=np.ones((1, 1)),
                   native_tp={"tp_m": np.ones((1, 2), dtype=bool), "tp": np.ones((1, 2), dtype=bool)})
        manifest = writer.close(["synthetic-image"])
        # The roles have different optima. Epoch 100 is one permanent best.
        high = 85 if role == "MASTER" else 100
        score = .8 - abs(epoch-high) / 1000
        return {"mask_map": score, "box_map": score, "prediction_manifest": {
            "directory": directory.name, "sha256": manifest["sha256"], "bytes": manifest["bytes"]}}

    def observe_best_v2(self, epoch, checkpoint, artifact_refs):
        selected = {}
        for role, model in (("MASTER", self.model), ("EMA", self.ema.ema)):
            evaluation = artifact_refs[role + "_STANDARD"]
            path = verify_receipt(self.root / "artifacts", evaluation)
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            value = wrapper["evaluation"]
            best = self.best.observe(epoch=epoch, role=role, metrics={k: value[k] for k in ("box_map", "mask_map")},
                state=native_state_dict(model), checkpoint=checkpoint, permanent=epoch in (100, 200), evaluation=evaluation,
                metadata={"evaluation_state_sha256": wrapper["binding"]["evaluation_state_sha256"]})
            selected[role] = relative_receipt(self.root / "artifacts", self.root / "artifacts" / best["record"])
        return selected

    def export_last_v2(self, root, repository, spec):
        record = repository.load(spec)
        roles = {"MASTER": record["state"], "EMA": record["ema"]["state"]}
        return {role: make_last(root / "artifacts", role,
                    {k: v for k, v in state.items() if not k.startswith(("state_head.", "region_projection."))},
                    {"checkpoint_sha256": spec["receipt"]["sha256"], "epoch": 200})
                for role, state in roles.items()}


def new_root(label):
    return OUTPUT / f"{label}_{uuid.uuid4().hex}"


def run(root, *, resume=False, failpoint=None, val_calls=None, crash=None, bindings=None, **kwargs):
    adapters = []
    def make(bound):
        adapter = SyntheticAdapter(root, bound, val_calls=[] if val_calls is None else val_calls, crash=crash)
        adapters.append(adapter)
        return adapter
    result = run_storage_cpu(make, profile=PROFILE, repository=REPO, output=root,
                             bindings=BINDINGS if bindings is None else bindings,
                             resume=resume, failpoint=failpoint, **kwargs)
    return result, adapters[-1]


def read_journal(root):
    info = json.loads((root / "owner.json").read_text(encoding="utf-8"))
    return StorageJournal(root / "journal.jsonl", info["bindings"], policy(200))


def at(event, step):
    def failpoint(actual, pos):
        if actual == event and pos == step:
            raise InjectedCrash((event, step))
    return failpoint


class StorageV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)

    def test_full_h200_uninterrupted_and_recovered_are_identical(self):
        control, recovery = new_root("control"), new_root("recovery")
        control_val, recovered_val = [], []
        with CPUExecutionGuard((OUTPUT,)) as guard:
            info, reference = run(control, val_calls=control_val)
            expected = _training_fingerprints(reference)
            self.assertEqual(reference.forward_calls, 4200)
            failpoints = [("after_update", 105), ("after_attempt_begin", 312),
                          ("after_checkpoint_commit", 420), ("after_validation_commit", 525),
                          ("after_update", random.Random(42).randrange(650, 900))]
            for index, (event, step) in enumerate(failpoints):
                with self.assertRaises(InjectedCrash):
                    run(recovery, resume=index > 0, val_calls=recovered_val, failpoint=at(event, step))
            recovered_info, repaired = run(recovery, resume=True, val_calls=recovered_val)
            observed = _training_fingerprints(repaired)
        self.assertEqual(expected, observed)
        self.assertEqual(info, recovered_info)
        self.assertEqual(info["permanent_checkpoints"], [0, 2100, 4200])
        self.assertEqual(control_val, recovered_val)
        self.assertEqual(len(recovered_val), 160)
        self.assertEqual(len(set(recovered_val)), 160)
        self.assertEqual(reference.counters, repaired.counters)
        self.assertEqual(guard.events["blocked_CUDA"], 0)
        journal = read_journal(recovery)
        seen = Counter(r["event"] for r in journal.rows)
        self.assertEqual(seen["CHECKPOINT_COMMIT"], 41)
        self.assertEqual(seen["RETIRE_COMMIT"], 38)
        self.assertEqual(seen["BEST_COMMIT"], 40)
        vals = [r for r in journal.val.values()]
        self.assertEqual(len(vals), 160)
        abandoned = [r["fields"]["step"]-r["fields"]["anchor"] for r in journal.rows if r["event"] == "SEGMENT_ABANDONED"]
        self.assertEqual(max(abandoned), 105)
        self.assertGreater(seen["UPDATE_EXECUTED"], 4200)
        accepted = [r for r in journal.rows if r["event"] == "UPDATE_EXECUTED"]
        projected = {r["fields"]["step"]: r["fields"]["replay_sha256"] for r in accepted}
        ref = {r["fields"]["step"]: r["fields"]["replay_sha256"] for r in read_journal(control).rows
               if r["event"] == "UPDATE_EXECUTED"}
        self.assertEqual(projected, ref)
        best = BestWeights(recovery / "artifacts").active()
        self.assertEqual([best[k]["epoch"] for k in ("MASTER", "EMA")], [85, 100])
        self.assertEqual(len(list((recovery / "artifacts").glob("best_weights_*.pt"))), 1)
        self.assertEqual(len(list((recovery / "checkpoints").glob("*/*.pt"))), 3)
        self.assertEqual(journal.identity["arm_id"], "a1_true")
        self.assertTrue(all(row["evaluator_sha256"] == BINDINGS["evaluator_sha256"] for row in vals))
        with self.assertRaisesRegex(ValueError, "completed run"):
            run(recovery, resume=True)
        cache = EpochPacketCache(recovery / "cache", json.loads((recovery / "owner.json").read_text())["bindings"],
                                 maximum_epoch_bytes=1024**2)
        cache_sizes = [sum(row["bytes"] for row in m["packets"]) for _p, m, _retired in cache._catalog()[0]]
        diagnostics = [row["bytes"] for row in journal.diagnostics.values()]
        val_sizes_full = []
        for event in vals:
            receipt = event["artifact"]
            doc = json.loads((recovery / "artifacts" / receipt["file"]).read_text(encoding="utf-8"))
            folder = recovery / "artifacts" / doc["evaluation"]["prediction_manifest"]["directory"]
            val_sizes_full.append(receipt["bytes"] + sum(p.stat().st_size for p in folder.iterdir()))
        active, checkpoint_peak = {}, 0
        for event in journal.rows:
            info = event["fields"]
            if event["event"] == "CHECKPOINT_COMMIT":
                active[info["step"]] = info["receipt"]["bytes"]
                checkpoint_peak = max(checkpoint_peak, sum(active.values()))
            elif event["event"] == "RETIRE_COMMIT":
                active.pop(info["step"])
        report = {
            "status": "PASS", "torch": torch.__version__, "logical_updates": 4200,
            "execution_attempt_updates": seen["UPDATE_EXECUTED"], "optimizer_reservations": seen["ATTEMPT_BEGIN"],
            "checkpoint_commit_count": seen["CHECKPOINT_COMMIT"], "retired_rolling_checkpoints": seen["RETIRE_COMMIT"],
            "maximum_recomputed_updates_in_test": max(abandoned), "mocked_VAL_passes": len(recovered_val),
            "native_VAL_passes": 0, "native_TEST_accesses": 0, "real_checkpoint_tensor_loads": 0,
            "native_forward_backward_updates": 0, "cuda_build": torch.version.cuda,
            "synthetic_sizes_bytes": {"checkpoints": [r["receipt"]["bytes"] for r in journal.checkpoints.values()],
                "journal_total": journal.bytes, "journal_rows": len(journal.rows),
                "journal_bytes_per_row_average": journal.bytes / len(journal.rows),
                "update_row_max": max(len(json.dumps(r, separators=(",", ":"), ensure_ascii=False).encode())+1 for r in accepted),
                "maximum_epoch_packets": max(cache_sizes),
                "two_live_plus_builder_cache_payload_peak": max(sum(cache_sizes[i:i+3]) for i in range(len(cache_sizes))),
                "checkpoint_payload_peak_during_rotation": checkpoint_peak,
                "diagnostic_maximum": max(diagnostics),
                "validation_per_pass_upper": max(val_sizes_full), "best_weight": best["MASTER"]["weights"]["bytes"],
                "last_exports": [p.stat().st_size for p in sorted((recovery / "artifacts").glob("last*"))]},
            "final_state_fingerprints": observed, "journal_sha256": sha256(recovery / "journal.jsonl"),
            "logical_trajectory_sha256": digest(projected), "journal_events": dict(seen),
            "durability_scope": "FILE_FSYNC_AND_WINDOWS_KERNEL_LEASE" if os.name == "nt" else "FILE_AND_DIRECTORY_FSYNC_AND_FLOCK",
        }
        if os.environ.get("EXP18_V2_MEASUREMENT"):
            write_json_exclusive(os.environ["EXP18_V2_MEASUREMENT"], report)

    def test_orphan_and_truncated_journal_hold(self):
        root = new_root("orphan")
        with self.assertRaises(InjectedCrash):
            run(root, failpoint=at("after_checkpoint_blob", 105))
        before = {p: sha256(p) for p in (root / "checkpoints").glob("*/*.pt")}
        self.assertEqual(len(before), 2)
        with self.assertRaisesRegex(ValueError, "Uncommitted"):
            run(root, resume=True)
        self.assertEqual({p: sha256(p) for p in before}, before)
        with (root / "journal.jsonl").open("ab") as stream:
            stream.write(b"{")
        with self.assertRaisesRegex(ValueError, "Torn"):
            read_journal(root)

    def test_interrupted_val_and_nonfinite_do_not_publish_a_false_commit(self):
        root = new_root("val")
        calls = []
        with self.assertRaises(InjectedCrash):
            run(root, failpoint=at("after_validation_begin", 105), val_calls=calls)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(ValueError, "Interrupted VAL"):
            run(root, resume=True, val_calls=calls)
        self.assertEqual(calls, [])
        root = new_root("nonfinite")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            run(root, crash="NONFINITE")
        self.assertEqual(read_journal(root).persisted, 0)
        self.assertEqual(read_journal(root).pending_step, 1)
        with self.assertRaisesRegex(ValueError, "existing HOLD"):
            run(root, resume=True)

    def test_amp_skip_and_validation_mutation_hold(self):
        root = new_root("amp_skip")
        with patch.object(torch.amp.GradScaler, "step", return_value=None):
            with self.assertRaisesRegex((ValueError, AssertionError), "No inf checks|AMP skipped"):
                run(root)
        self.assertEqual(read_journal(root).persisted, 0)
        self.assertTrue(list(root.glob("hold_*.json")))
        root = new_root("val_mutation")
        with self.assertRaisesRegex(ValueError, "modified protected training"):
            run(root, crash="VAL_MUTATION")
        journal = read_journal(root)
        self.assertEqual(journal.persisted, 105)
        self.assertEqual(journal.pending_val, (105, "MASTER_STANDARD"))

    def test_corrupt_checkpoint_and_binding_hold_before_resume(self):
        root = new_root("corrupt")
        with self.assertRaises(InjectedCrash):
            run(root, failpoint=at("after_checkpoint_commit", 105))
        with self.assertRaisesRegex(ValueError, "identity"):
            run(root, resume=True, bindings={**BINDINGS, "data_sha256": "d"*64})
        initial = read_journal(root).checkpoints[0]
        path = root / "checkpoints" / initial["generation"] / initial["receipt"]["file"]
        with path.open("ab") as stream:
            stream.write(b"fault")
        with self.assertRaisesRegex(ValueError, "size or SHA"):
            run(root, resume=True)

    def test_fsync_failure_keeps_old_checkpoint_and_orphan(self):
        root = new_root("fsync")
        from tools.exp18.checkpoint import ImmutableCheckpointStore
        original_save = ImmutableCheckpointStore.save
        def save_bound(store, record):
            if record["position"]["global_update"] == 105:
                with patch("tools.exp18.checkpoint.os.fsync", side_effect=OSError("flush failure")):
                    return original_save(store, record)
            return original_save(store, record)
        with patch.object(ImmutableCheckpointStore, "save", save_bound):
            with self.assertRaisesRegex(OSError, "flush failure"):
                run(root)
        self.assertEqual(read_journal(root).persisted, 0)
        self.assertEqual(len(list((root / "checkpoints").glob("*/*.pt"))), 2)

    def test_repeat_skip_budget_and_chain_rejected(self):
        for attempted in (1, 3, 107):
            root = new_root("bad_cursor")
            with self.assertRaises(InjectedCrash):
                run(root, failpoint=at("after_update", 1))
            j = read_journal(root)
            with self.assertRaisesRegex(ValueError, "out of order or over budget"):
                j.append("ATTEMPT_BEGIN", step=attempted, attempt=j.run_attempt)
            self.assertEqual(read_journal(root).executed, 1)
        root = new_root("chain")
        with self.assertRaises(InjectedCrash):
            run(root, failpoint=at("after_update", 1))
        with (root / "journal.jsonl").open("ab") as stream:
            stream.write((root / "journal.jsonl").read_bytes().splitlines(keepends=True)[-1])
        with self.assertRaisesRegex(ValueError, "chain"):
            read_journal(root)

    def test_cache_rebuild_equivalence_and_reject_altered_content(self):
        root = new_root("packets")
        root.mkdir()
        cache = EpochPacketCache(root, BINDINGS, maximum_epoch_bytes=1024**2)
        payload = [{"i": i} for i in range(21)]
        accepted = [digest(row) for row in payload]
        def build(indent):
            return lambda: ((json.dumps(row, indent=indent).encode(), key) for row, key in zip(payload, accepted, strict=True))
        def fingerprint(wire):
            return digest(json.loads(wire))
        old, first = cache.open_or_build(0, accepted, build(None), fingerprint=fingerprint)
        cache.open_or_build(1, accepted, build(None), fingerprint=fingerprint)
        cache.open_or_build(2, accepted, build(None), fingerprint=fingerprint)
        self.assertTrue((old / "retire_commit.json").exists())
        self.assertFalse(list(old.glob("*.packet")))
        new, rebuilt = cache.open_or_build(0, accepted, build(2), fingerprint=fingerprint)
        self.assertNotEqual(old, new)
        for a, b in zip(first["packets"], rebuilt["packets"], strict=True):
            self.assertEqual(a["content_sha256"], b["content_sha256"])
            self.assertNotEqual(a["sha256"], b["sha256"])
        with self.assertRaisesRegex(ValueError, "content SHA"):
            cache.open_or_build(3, accepted, lambda: [(b"{}", accepted[0])], fingerprint=fingerprint)
        self.assertEqual(len(list(new.glob("*.packet"))), 21)

    def test_profile_and_single_writer(self):
        for horizon in (100, 200):
            runtime, scientific = load_storage_profile(REPO / f"configs/exp18/runtime_storage_v2_h{horizon}.json", REPO)
            self.assertEqual(runtime["storage"]["permanent_epochs"], [0, horizon//2, horizon])
            self.assertEqual(scientific["train"]["updates"], horizon * 21)
            self.assertTrue(all(not value for value in runtime["budgets"].values()))
        root = new_root("lease")
        root.mkdir()
        with lease(root, create=True):
            with self.assertRaises(OSError):
                with lease(root, create=False):
                    self.fail("concurrent lease opened")


if __name__ == "__main__":
    unittest.main()
