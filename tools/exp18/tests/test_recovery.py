import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest
import uuid
import warnings
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from tools.exp18.checkpoint import (ImmutableCheckpointStore, capture_training_state,
                                    restore_training_state, state_hash, validate_payload,
                                    backend_state, restore_backend)
from tools.exp18.execution import AppendOnlyJournal, ExperimentExecutor, SyntheticExecutor, run_native
from tools.exp18.model import add_state_head, native_state_dict
from tools.exp18.optimizer import CPUModelEMA, HeadAwareEMA, build_adamw, joint_gradient_stats
from tools.exp18.protocol import make_protocol
from tools.exp18.native_adapter import NativeAdapter, RetainedRunLease, check_append_only_capacity
from tools.exp18.checkpoint import SCHEMA, file_sha

BINDINGS = {"code_sha256": "a" * 64, "protocol_sha256": "b" * 64, "matching_sha256": "c" * 64}


class RecoveryToy(nn.Module):
    exp18_synthetic = True
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 64)
        self.bn = nn.BatchNorm1d(64)
        self.native_head = nn.Linear(64, 1)
        self.region_projection = nn.Linear(64, 4)
        add_state_head(self, seed=180042)

    def loss(self, x, *, active):
        raw = self.bn(self.backbone(x))
        result = self.native_head(raw).square().mean() + 8 * self.region_projection(raw).square().mean()
        if active:
            result = result + torch.nn.functional.softplus(-self.state_head(raw)).mean()
        return result


def make(seed=15):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = RecoveryToy().train()
    optimizer, names = build_adamw(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: 1 - .01 * index)
    ema = HeadAwareEMA(CPUModelEMA(model))
    scaler = torch.amp.GradScaler("cpu", init_scale=1.)
    return SyntheticExecutor(model, optimizer, scheduler, ema), names, scaler


def advance(engine, active):
    factor = .5 + random.random() + float(np.random.random())
    x = torch.randn(4, 4) * factor
    row = engine.step(engine.model.loss(x, active=active), head_active=active)
    engine.scheduler.step()
    return row


def capture(engine, scaler):
    update = engine.global_update
    return capture_training_state(engine.model, engine.optimizer, engine.scheduler, engine.ema,
        bindings=BINDINGS, position={"epoch": 0, "batch_position": update, "global_update": update},
        sampler_state={"next_batch": update, "cache_manifest_sha256": "d" * 64},
        head_active_updates=engine.head_active_updates, scaler=scaler,
        counters={"optimizer_attempts": update, "amp_skips": 0}, extra={"matching_sha256": "e" * 64})


class RecoveryTests(unittest.TestCase):
    def test_empty_active_empty_active_full_recovery(self):
        sequence = [False, True, False, True]
        reference, _, scaler = make()
        expected = []
        for active in sequence:
            advance(reference, active)
            expected.append(capture(reference, scaler))
        engine, names, scaler = make()
        root = Path(__file__).resolve().parents[3] / "private_assets" / "exp18_tests" / uuid.uuid4().hex
        store = ImmutableCheckpointStore(root, BINDINGS, create=True)
        before_empty = None
        for index, active in enumerate(sequence):
            if index == 2:
                before_empty = (engine.model.state_head.weight.detach().clone(),
                                copy.deepcopy(engine.optimizer.state[engine.model.state_head.weight]),
                                engine.ema.ema.state_head.weight.detach().clone())
            advance(engine, active)
            if index == 0:
                self.assertNotIn(engine.model.state_head.weight, engine.optimizer.state)
            if index == 2:
                self.assertTrue(torch.equal(before_empty[0], engine.model.state_head.weight))
                self.assertEqual(state_hash(before_empty[1]), state_hash(engine.optimizer.state[engine.model.state_head.weight]))
                self.assertTrue(torch.equal(before_empty[2], engine.ema.ema.state_head.weight))
            record = capture(engine, scaler)
            self.assertEqual(state_hash(record), state_hash(expected[index]))
            receipt = store.save(record)
            with self.assertRaisesRegex(ValueError, "already exists"):
                store.save(record)
            loaded = store.load(receipt)
            recovered, recovered_names, recovered_scaler = make(seed=900 + index)
            state = restore_training_state(recovered.model, recovered.optimizer, recovered.scheduler,
                recovered.ema, loaded, bindings=BINDINGS, scaler=recovered_scaler)
            recovered.global_update = state["position"]["global_update"]
            recovered.head_active_updates = state["head_active_updates"]
            self.assertEqual(recovered_names, names)
            self.assertEqual(state_hash(capture(recovered, recovered_scaler)), state_hash(record))
            engine, scaler = recovered, recovered_scaler
        self.assertEqual(engine.global_update, 4)
        self.assertEqual(engine.head_active_updates, 2)
        self.assertEqual(float(engine.optimizer.state[engine.model.state_head.weight]["step"]), 2.)
        self.assertEqual(len(list(root.glob("*.pt"))), 4)
        self.assertEqual(len(list(root.glob("*.commit.json"))), 4)

    def test_head_ema_freezes_while_native_ema_is_exact(self):
        engine, _, _ = make()
        direct = CPUModelEMA(engine.model)
        for active in (False, True, False, True):
            before = engine.ema.ema.state_head.weight.detach().clone()
            advance(engine, active)
            direct.update(engine.model)
            for name, value in direct.ema.state_dict().items():
                if not name.startswith("state_head."):
                    self.assertTrue(torch.equal(value, engine.ema.ema.state_dict()[name]))
            if not active:
                self.assertTrue(torch.equal(before, engine.ema.ema.state_head.weight))
        self.assertEqual(direct.updates, engine.ema.updates)

    def test_groups_and_native_export_filter(self):
        engine, names, _ = make()
        self.assertIn("state_head.weight", names[0])
        self.assertEqual(sum(group.count("state_head.weight") for group in names), 1)
        self.assertEqual([g["param_group"] for g in engine.optimizer.param_groups], ["weight", "bn", "bias"])
        self.assertEqual([g["weight_decay"] for g in engine.optimizer.param_groups], [.0005, 0., 0.])
        state = engine.model.state_dict()
        self.assertIn("state_head.weight", state)
        self.assertIn("region_projection.weight", state)
        self.assertNotIn("state_head.weight", native_state_dict(engine.model))
        self.assertNotIn("region_projection.weight", native_state_dict(engine.model))

    def test_counter_and_binding_tampering_rejected(self):
        engine, _, scaler = make()
        advance(engine, True)
        record = capture(engine, scaler)
        wrong = copy.deepcopy(record)
        wrong["head_active_updates"] = 0
        with self.assertRaisesRegex(ValueError, "EMA clocks"):
            validate_payload(wrong, BINDINGS)
        wrong = copy.deepcopy(BINDINGS)
        wrong["matching_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "binding changed"):
            validate_payload(record, wrong)
        wrong = copy.deepcopy(record)
        for group, names in zip(wrong["optimizer"]["param_groups"], wrong["optimizer_names"], strict=True):
            for identifier, name in zip(group["params"], names, strict=True):
                if name == "state_head.weight":
                    wrong["optimizer"]["state"][identifier]["step"] = torch.tensor(100.)
        with self.assertRaisesRegex(ValueError, "clock differs"):
            validate_payload(wrong, BINDINGS)

    def test_backend_policy_is_restored_with_full_state(self):
        original = backend_state()
        try:
            engine, _, scaler = make()
            advance(engine, False)
            record = capture(engine, scaler)
            torch.backends.cudnn.benchmark = not original["cudnn_benchmark"]
            torch.use_deterministic_algorithms(not original["deterministic_algorithms"], warn_only=True)
            recovered, _, recovered_scaler = make(seed=88)
            restore_training_state(recovered.model, recovered.optimizer, recovered.scheduler, recovered.ema,
                                   record, bindings=BINDINGS, scaler=recovered_scaler)
            self.assertEqual(backend_state(), record["backend"])
        finally:
            restore_backend(original)

    def test_ema_requires_fresh_matching_activity(self):
        engine, _, _ = make()
        with self.assertRaisesRegex(ValueError, "explicitly set"):
            engine.ema.update(engine.model)

    def test_stale_journal_is_rejected_before_attempt_reservation(self):
        root = Path(__file__).resolve().parents[3] / "private_assets" / "exp18_tests" / uuid.uuid4().hex
        first = AppendOnlyJournal(root / "journal.jsonl", BINDINGS, create=True)
        stale = AppendOnlyJournal(root / "journal.jsonl", BINDINGS)
        first.append("INITIAL_STATE_COMMIT", checkpoint_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "Stale journal"):
            stale.append("ATTEMPT_BEGIN", step=1)
        self.assertIsNone(stale.pending)
        fresh = AppendOnlyJournal(root / "journal.jsonl", BINDINGS)
        self.assertEqual(len(fresh.rows), 1)

    def test_domain_gradient_stats_clip_and_empty_head(self):
        engine, _, _ = make()
        for name, parameter in engine.model.named_parameters():
            if parameter.requires_grad and not name.startswith("state_head."):
                parameter.grad = torch.full_like(parameter, 3.)
        saved = {name: None if p.grad is None else p.grad.clone() for name, p in engine.model.named_parameters()}
        before = joint_gradient_stats(engine.model)
        self.assertGreater(before["main"]["norm"], 10.)
        self.assertEqual(before["main"]["missing_tensors"], 0)
        self.assertEqual(before["state_head"]["missing_tensors"], 1)
        self.assertEqual(before["state_head"]["norm"], 0.)
        for name, parameter in engine.model.named_parameters():
            self.assertTrue(parameter.grad is None if saved[name] is None else torch.equal(saved[name], parameter.grad))
        torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 10.)
        after = joint_gradient_stats(engine.model)
        self.assertLessEqual(after["joint_norm"], 10.00001)
        self.assertEqual(after["main"]["nonfinite_elements"], 0)
        self.assertIsNone(engine.model.state_head.weight.grad)
        engine.model.state_head.weight.grad = torch.zeros_like(engine.model.state_head.weight)
        present_zero = joint_gradient_stats(engine.model)["state_head"]
        self.assertEqual(present_zero["missing_tensors"], 0)
        self.assertEqual(present_zero["present_tensors"], 1)
        self.assertEqual(present_zero["zero_elements"], 64)

    def test_native_factory_rejected_before_construction(self):
        called = []
        with self.assertRaises(ValueError):
            run_native(lambda: called.append(True), make_protocol(), BINDINGS, None, None)
        self.assertEqual(called, [])
        with patch("tools.exp18.native_adapter.native_trainer_factory") as native_factory:
            with self.assertRaises(ValueError):
                NativeAdapter({}, make_protocol(), BINDINGS, "/unused", arm="j1", evaluator=None, exporter=None)
            native_factory.assert_not_called()

    def test_append_only_capacity_includes_every_epoch_and_checkpoint(self):
        resource = {"epoch_pack_bytes": 100, "recovery_bytes": 50, "reserve_bytes": 1000}
        expected = 200 * 100 + (4200 + 200 + 1) * 50 + 1000
        with patch("tools.exp18.native_adapter.shutil.disk_usage", return_value=SimpleNamespace(free=expected)):
            self.assertEqual(check_append_only_capacity(Path(__file__).parent, make_protocol(), resource), expected)
        with patch("tools.exp18.native_adapter.shutil.disk_usage", return_value=SimpleNamespace(free=expected - 1)):
            with self.assertRaisesRegex(ValueError, "insufficient"):
                check_append_only_capacity(Path(__file__).parent, make_protocol(), resource)

    @unittest.skipUnless(os.name == "posix", "Native flock backend requires Linux")
    def test_retained_lease_excludes_other_fds_and_processes_and_survives_failure(self):
        root = Path(__file__).resolve().parents[3] / "private_assets" / "exp18_tests" / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        path = root / "run.lock"
        first = RetainedRunLease(path, BINDINGS, create=True)
        saved = path.read_bytes()
        script = """import json,sys
from tools.exp18.native_adapter import RetainedRunLease
try:
    lease=RetainedRunLease(sys.argv[1],json.loads(sys.argv[2]))
except ValueError as exc:
    print(str(exc));sys.exit(23)
else:
    lease.close()
"""
        command = [sys.executable, "-B", "-S", "-c", script, str(path), json.dumps(BINDINGS)]
        try:
            with self.assertRaisesRegex(ValueError, "Another process/FD"):
                RetainedRunLease(path, BINDINGS)
            blocked = subprocess.run(command, cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, check=False)
            self.assertEqual(blocked.returncode, 23, blocked.stderr)
            self.assertIn("Another process/FD", blocked.stdout)
        finally:
            first.close()
        self.assertEqual(path.read_bytes(), saved)
        acquired = subprocess.run(command, cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, check=False)
        self.assertEqual(acquired.returncode, 0, acquired.stderr)
        with self.assertRaisesRegex(RuntimeError, "injected setup"):
            with RetainedRunLease(path, BINDINGS):
                raise RuntimeError("injected setup")
        with RetainedRunLease(path, BINDINGS) as final:
            self.assertTrue(final.held)
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), saved)

    def test_uncommitted_orphan_and_torn_receipt_are_preserved_and_rejected(self):
        engine, _, scaler = make()
        advance(engine, True)
        record = capture(engine, scaler)
        root = Path(__file__).resolve().parents[3] / "private_assets" / "exp18_tests" / uuid.uuid4().hex
        store = ImmutableCheckpointStore(root, BINDINGS, create=True)
        with patch("tools.exp18.checkpoint.torch.load", side_effect=RuntimeError("injected readback interruption")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                store.save(record)
        path = root / "update_000001_r0.pt"
        self.assertTrue(path.is_file())
        self.assertFalse(path.with_suffix(".commit.json").exists())
        receipt = {"schema": SCHEMA, "owner_id": store.owner["owner_id"], "file": path.name,
                   "global_update": 1, "revision": 0, "artifact_kind": "synthetic_cpu",
                   "bytes": path.stat().st_size, "sha256": file_sha(path),
                   "state_sha256": state_hash(record), "bindings_sha256": state_hash(BINDINGS)}
        with self.assertRaisesRegex(ValueError, "receipt missing"):
            store.load(receipt)
        with self.assertRaisesRegex(ValueError, "already exists"):
            store.save(record)
        # A new truncated receipt simulates a power interruption; no old file is changed.
        with path.with_suffix(".commit.json").open("x", encoding="utf-8") as stream:
            stream.write("{")
        with self.assertRaises(ValueError):
            store.load(receipt)
        self.assertEqual(path.with_suffix(".commit.json").read_text(), "{")
        self.assertEqual(file_sha(path), receipt["sha256"])

    def test_protocol_event_machine_complete_budget_and_callbacks(self):
        class Tiny(nn.Module):
            exp18_synthetic = True
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([.2]))
                self.region_projection = nn.Linear(1, 1)
                add_state_head(self, seed=180042)

        class Cursor:
            def __init__(self, epoch, index=0):
                self.epoch, self.next_index = epoch, index
            def next_batch(self):
                self.next_index += 1
                return None
            def state_dict(self):
                return {"epoch": self.epoch, "next_batch": self.next_index}

        class Adapter:
            def __init__(self):
                self.model = Tiny()
                self.optimizer, _ = build_adamw(self.model)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
                self.ema = HeadAwareEMA(CPUModelEMA(self.model))
                self.scaler = None
                self.evaluations, self.starts = [], []
            def start_epoch(self, epoch):
                self.starts.append(epoch)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.scheduler.step()
            def build_cache(self, epoch):
                return Cursor(epoch)
            def open_cache(self, sampler):
                return Cursor(sampler["epoch"], sampler["next_batch"])
            def before_batch(self, step):
                pass
            def forward_batch(self, batch, step):
                state = self.model.weight.new_zeros(())
                active = step % 2 == 0
                if active:
                    state = self.model.state_head.weight.square().sum()
                loss = self.model.weight.square().sum() + self.model.region_projection(self.model.weight).square().sum() + state
                return SimpleNamespace(total=loss, state=state, head_active=active, matching_sha256="e" * 64)
            def capture(self, *, position, sampler_state, head_active_updates, extra):
                return capture_training_state(self.model, self.optimizer, self.scheduler, self.ema,
                    bindings=BINDINGS, position=position, sampler_state=sampler_state,
                    head_active_updates=head_active_updates, extra=extra)
            def restore(self, record):
                return restore_training_state(self.model, self.optimizer, self.scheduler, self.ema, record, bindings=BINDINGS)
            def evaluate(self, epoch, role):
                self.evaluations.append((epoch, role))
            def export(self, role):
                return {"role": role, "native_keys": sorted(native_state_dict(self.model))}

        class MemoryStore:
            def __init__(self):
                self.positions = set()
            def save(self, record):
                position = record["position"]
                key = (position["global_update"], position["revision"])
                if key in self.positions:
                    raise ValueError("Immutable position repeated")
                self.positions.add(key)
                self.latest = copy.deepcopy(record)
                return {"sha256": state_hash(record)}

        # The full H200/4200 event machine runs on a tiny synthetic CPU graph.
        root = Path(__file__).resolve().parents[3] / "private_assets" / "exp18_tests" / uuid.uuid4().hex
        journal = AppendOnlyJournal(root / "journal.jsonl", BINDINGS, create=True)
        adapter, store = Adapter(), MemoryStore()
        executor = ExperimentExecutor(adapter, make_protocol(200), BINDINGS, store, journal, synthetic=True)
        stopped = executor.run(stop_after=7)
        self.assertEqual(stopped["updates"], 7)
        starts = adapter.starts.copy()
        adapter = Adapter()
        journal = AppendOnlyJournal(root / "journal.jsonl", BINDINGS)
        executor = ExperimentExecutor(adapter, make_protocol(200), BINDINGS, store, journal, synthetic=True)
        result = executor.run(recovered=store.latest)
        self.assertEqual(result["updates"], 4200)
        self.assertEqual(executor.head_active_updates, 2100)
        self.assertEqual(starts + adapter.starts, list(range(200)))
        self.assertEqual(adapter.evaluations, [(e, role) for e in range(5, 201, 5) for role in ("MASTER", "EMA")])
        self.assertEqual(len(store.positions), 4401)
        self.assertEqual(journal.committed, 4200)
        with self.assertRaisesRegex(ValueError, "cannot restart"):
            executor.run()
        reopened = AppendOnlyJournal(root / "journal.jsonl", BINDINGS)
        self.assertEqual(reopened.committed, 4200)


if __name__ == "__main__":
    unittest.main()
