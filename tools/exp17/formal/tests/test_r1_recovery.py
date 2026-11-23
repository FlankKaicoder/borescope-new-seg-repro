"""Bounded CPU synthetic R1 transactions; no research data or native VAL."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tools.exp17.formal.authorization import HardGate, RunJournal, digest, exclusive_json, file_sha
from tools.exp17.formal.checkpoint import AtomicCheckpointStore, state_hash
from tools.exp17.formal.executor import run_session
from tools.exp17.formal.monitor import epoch_summary
from tools.exp17.formal.rng import capture_rng, restore_rng
from tools.exp17.formal.sampler import SealedEpochCache
from tools.exp17.formal import repair_r1 as r1
from tools.exp17.formal.tests.test_executor import SyntheticAdapter, PROTOCOL, session
from tools.exp17.formal.tests.test_recovery import BINDINGS, seed_all, synthetic_epoch
from tools.exp17.formal.validation import METRIC_KEYS


def record(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def fixture_evaluation():
    metrics = {key: .2 for key in METRIC_KEYS}; metrics["fitness"] = .4
    return {"role": "master", "standard": {"overall": metrics}, "fixed": {"fixture": True}, "fitness": .4}


class R1Adapter(SyntheticAdapter):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.config = {"val_ledger_root": str(output / "val_passes"), "val_output": str(output / "val")}
        for key in self.config:
            Path(self.config[key]).mkdir(exist_ok=True)
        self.sampler = None
    def set_sampler(self, cursor):
        self.sampler = cursor; super().set_sampler(cursor)
    def evaluate(self, epoch, role):
        if epoch == 5:
            assert role == "ema"
            self.val_events.append((epoch, role))
            result = fixture_evaluation(); result["role"] = role
            for mode in ("standard", "fixed"):
                exclusive_json(Path(self.config["val_ledger_root"]) / f"epoch005_ema_{mode}.json", {
                    "epoch":5,"role":"ema","mode":mode,"protocol_sha256":r1.PROTOCOL_SHA,
                    "state":"ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY"})
                folder = Path(self.config["val_output"]) / f"epoch005_ema_{mode}"
                folder.mkdir(); exclusive_json(folder / "evaluation.json", result[mode])
            return result
        return super().evaluate(epoch, role)
    def build_cache(self, epoch, directory):
        self.cache_epochs.append(epoch)
        return SealedEpochCache.build(directory, synthetic_epoch(epoch), epoch=epoch, bindings=self.bindings,
            batch_fingerprint=lambda n,b:{"index":n,"tensor_and_pair_identity":state_hash(b)}, expected_batches=21)


class R1RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        seed_all(); self.first = SyntheticAdapter()
        self.output, self.journal, self.old_store = session(self.root / "run")
        run_session(self.first, PROTOCOL, BINDINGS, self.output, self.journal, self.old_store,
                    synthetic_stop_after=105)
        self.old, self.receipt = self.old_store.load_latest()
        self.journal.append("AUX_BEGIN", epoch=5)
        rows = [json.loads(x) for x in (self.output / "steps.jsonl").read_text().splitlines()]
        exclusive_json(self.output / "epoch005.json", epoch_summary(5, rows[-21:], ema_updates=105))
        self.journal.append("HARD_GATE", error_type="IsolationViolation", message="CPU_SYNTHETIC_FIXTURE")
        self.origin_journal = record(self.journal.path)
        self.origin_journal.update(rows=221, last_row_sha256=self.journal.rows[-1]["sha256"])
        self.d = {"origin": {"execution_id": "CPU_SYNTHETIC_R1_ONLY", "journal": self.origin_journal,
                   "checkpoint": record(self.old_store.directory / self.receipt["file"]), "output": str(self.output)},
                  "legacy_caches": [{"directory": str(self.output / "epoch_cache" / f"epoch_{e:03d}"),
                    "manifest_sha256": file_sha(self.output / "epoch_cache" / f"epoch_{e:03d}" / "manifest.json")}
                    for e in (3, 4)]}
        self.d_sha = "d" * 64
        self.b1 = r1.effective_bindings(BINDINGS, self.d_sha, "e" * 64)
        self.patch_sha = patch.object(r1, "CHECKPOINT_SHA", self.receipt["sha256"])
        self.patch_journal = patch.object(r1, "JOURNAL_SHA", self.origin_journal["sha256"])
        self.patch_caches = patch.object(r1, "LEGACY_MANIFESTS", {int(Path(r["directory"]).name.split("_")[1]):
                                         r["manifest_sha256"] for r in self.d["legacy_caches"]})
        self.patch_sha.start(); self.patch_journal.start()
        self.patch_caches.start(); self.addCleanup(self.patch_caches.stop)
        self.addCleanup(self.patch_sha.stop); self.addCleanup(self.patch_journal.stop)
        self.store = AtomicCheckpointStore(self.output / "recovery_r1_v1", self.b1)
        self.association = {"master_evaluation": fixture_evaluation(), "association_type": "CPU_SYNTHETIC_ONLY"}

    def recover(self, adapter=None):
        adapter = adapter or R1Adapter(self.output)
        journal = r1.R1Journal(self.journal.path, self.d, self.d_sha)
        with patch("tools.exp17.formal.repair_evidence.verify_historical_master", return_value=self.association), \
             patch("tools.exp17.formal.repair_evidence.validate_evaluation_pair", return_value=.4):
            payload = r1.resume_epoch5(adapter, PROTOCOL, self.d, BINDINGS, self.b1,
                self.output, journal, self.old_store, self.store, "a" * 64)
        return payload, adapter, journal

    def test_pending_default_rejects_and_exact_r1_closes_without_updates(self):
        with self.assertRaises(HardGate): RunJournal(self.journal.path)
        original_bytes = self.journal.path.read_bytes()
        payload, adapter, journal = self.recover()
        self.assertEqual(adapter.val_events, [(5, "ema")])
        self.assertEqual(adapter.diagnostic_epochs, [])
        self.assertEqual(journal.committed, 105); self.assertIsNone(journal.pending_aux)
        self.assertTrue(payload["extra"]["epoch_events_complete"])
        r1.require_same_training_state(self.old, payload)
        self.assertTrue(self.journal.path.read_bytes().startswith(original_bytes))
        self.assertEqual(file_sha(self.old_store.directory / self.receipt["file"]), self.receipt["sha256"])
        self.assertEqual(RunJournal(self.journal.path).committed, 105)
        self.assertTrue(r1.R1Journal(self.journal.path, self.d, self.d_sha).r1_closed)
        with self.assertRaises(ValueError): AtomicCheckpointStore(self.store.directory, BINDINGS).load_latest()
        with self.assertRaises(ValueError): AtomicCheckpointStore(self.old_store.directory, self.b1).load_latest()

    def test_update106_equals_direct_and_scheduler_only_advances_once(self):
        payload, resumed, journal = self.recover()
        resumed.bindings = self.b1
        resolver = r1.legacy_cache_resolver(self.d, BINDINGS, self.b1)
        run_session(resumed, PROTOCOL, self.b1, self.output, journal, self.store,
                    recovered=payload, synthetic_stop_after=106, legacy_cache_bindings=resolver)
        actual, _ = self.store.load_latest()
        # Same initial state and batch; no AUX work may change training RNG.
        reference = SyntheticAdapter()
        from tools.exp17.formal.checkpoint import restore_training_state
        restore_training_state(reference.trainer, self.old, bindings=BINDINGS)
        reference.start_epoch(5)
        batch = synthetic_epoch(5)[0]
        reference.cursor = type("Cursor", (), {"next_index": 1})()
        reference.train_batch(batch, 106)
        self.assertEqual(state_hash(reference.trainer.model.state_dict()), actual["master_state_sha256"])
        self.assertEqual(state_hash(reference.trainer.optimizer.state_dict()), state_hash(actual["optimizer"]))
        self.assertEqual(reference.trainer.scheduler.last_epoch, self.old["scheduler"]["last_epoch"] + 1)
        self.assertEqual(reference.trainer.scheduler.last_epoch, actual["scheduler"]["last_epoch"])
        self.assertEqual(actual["position"], {"epoch":5,"batch_position":1,"global_update":106})
        self.assertEqual(resumed.cache_epochs, [5])
        self.assertFalse((self.output / "epoch_cache" / "epoch_003").exists())

    def test_wrong_prefix_aux_or_attempt_never_admitted(self):
        changed = copy.deepcopy(self.d); changed["origin"]["journal"]["sha256"] = "0" * 64
        with self.assertRaises(HardGate): r1.R1Journal(self.journal.path, changed, self.d_sha)
        journal = r1.R1Journal(self.journal.path, self.d, self.d_sha)
        with self.assertRaises(HardGate): journal.begin(106)
        with self.assertRaises(HardGate): journal.begin(2101)

    def test_claimed_ema_without_completion_is_hold_no_eval(self):
        adapter = R1Adapter(self.output)
        exclusive_json(Path(adapter.config["val_ledger_root"]) / "epoch005_ema_standard.json", {"fixture":True})
        with self.assertRaisesRegex(HardGate, "no accepted completion"):
            self.recover(adapter)
        self.assertEqual(adapter.val_events, [])
        self.assertFalse((self.store.directory / "latest.json").exists())

    def test_partial_best_is_hold_not_repeated_val(self):
        # Stop after EMA's trusted completion but before Best publication.
        with patch("tools.exp17.formal.exports.save_best_bundle", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"): self.recover()
        (self.output / "best").mkdir(exist_ok=True)
        exclusive_json(self.output / "best" / "best_master.json", {"partial":True})
        adapter = R1Adapter(self.output)
        with self.assertRaisesRegex(HardGate, "Partial Best"):
            self.recover(adapter)
        self.assertEqual(adapter.val_events, [])

    def test_checkpoint_ready_before_aux_commit_can_reuse_receipts(self):
        original_append = RunJournal.append
        def interrupt(journal, event, **values):
            if event == "AUX_COMMIT" and values.get("epoch") == 5:
                raise RuntimeError("before aux commit")
            return original_append(journal, event, **values)
        with patch.object(RunJournal, "append", interrupt):
            with self.assertRaisesRegex(RuntimeError, "before aux commit"): self.recover()
        adapter = R1Adapter(self.output)
        payload, _, journal = self.recover(adapter)
        self.assertEqual(adapter.val_events, [])
        self.assertEqual(journal.committed, 105); self.assertIsNone(journal.pending_aux)
        self.assertTrue(payload["extra"]["epoch_events_complete"])

    def test_ema_receipt_cannot_omit_its_five_bound_artifacts(self):
        with patch("tools.exp17.formal.exports.save_best_bundle", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"): self.recover()
        path = self.output / "r1_provenance" / "ema_role_completion.json"
        receipt = json.loads(path.read_text()); receipt["artifacts"] = []
        path.write_text(json.dumps(receipt))
        adapter = R1Adapter(self.output)
        with self.assertRaisesRegex(HardGate, "exact five artifacts"):
            self.recover(adapter)
        self.assertEqual(adapter.val_events, [])

    def test_legacy_whitelist_and_nonapproved_future_cache(self):
        resolver = r1.legacy_cache_resolver(self.d, BINDINGS, self.b1)
        item = self.d["legacy_caches"][0]
        self.assertEqual(resolver(item["directory"], item["manifest_sha256"]), BINDINGS)
        with self.assertRaises(HardGate): resolver(item["directory"], "0" * 64)
        self.assertEqual(resolver(self.output / "epoch_cache" / "epoch_005", "0" * 64), self.b1)
        extra = copy.deepcopy(self.d); extra["legacy_caches"].append({"directory":"/tmp/other","manifest_sha256":"0"*64})
        with self.assertRaises(HardGate): r1.legacy_cache_resolver(extra, BINDINGS, self.b1)

    def test_training_state_mutation_cannot_be_migration(self):
        changed = copy.deepcopy(self.old); changed["audit"]["effective_updates"] += 1
        with self.assertRaises(HardGate): r1.require_same_training_state(self.old, changed)
        changed = copy.deepcopy(self.old); changed["rng"]["torch_cpu"][0] ^= 1
        with self.assertRaises(HardGate): r1.require_same_training_state(self.old, changed)


if __name__ == "__main__": unittest.main()
