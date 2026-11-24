"""CPU-only strict-state and bounded continuation tests for the E10 addendum."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import observation_extension as extension
from tools.exp17.formal.authorization import HardGate, RunJournal, digest, exclusive_json, file_sha
from tools.exp17.formal.checkpoint import (AtomicCheckpointStore, capture_training_state,
    restore_training_state, state_hash, validate_payload)
from tools.exp17.formal.executor import run_session
from tools.exp17.formal.r5_gate import FOCUS, RULE, decide
from tools.exp17.formal.sampler import SealedEpochCache
from tools.exp17.formal.tests.test_executor import PROTOCOL, SyntheticAdapter, session
from tools.exp17.formal.tests.test_recovery import BINDINGS, seed_all, synthetic_epoch

ADDENDUM_SHA = "e" * 64
NEW_CODE_SHA = "f" * 64
B0 = copy.deepcopy(BINDINGS)


def reports(gap=1.):
    return [{"batch_position": p, "bn_mean_gap": {name: {"D": gap} for name in FOCUS}} for p in RULE["positions"]]


def decision(ap=.2, gap=1., artifact=None):
    value = decide(reports(gap), {"standard": {"overall": {"metrics/mAP50-95(M)": ap}}})
    value["artifacts"] = {str(artifact.resolve()): file_sha(artifact)} if artifact else {}
    return value


class Adapter(SyntheticAdapter):
    """One tiny CPU network with explicit B0/B1 sealed-cache metadata."""
    def __init__(self, bindings):
        super().__init__()
        self.bindings = bindings

    def build_cache(self, epoch, directory):
        self.cache_epochs.append(epoch)
        return SealedEpochCache.build(directory, synthetic_epoch(epoch), epoch=epoch, bindings=self.bindings,
            batch_fingerprint=lambda n, batch: {"index": n, "tensor_and_pair_identity": state_hash(batch)}, expected_batches=21)

    def gradient_diagnostics(self):
        self.diagnostic_epochs.append(self.trainer.epoch + 1)
        return reports()

    def evaluate(self, epoch, role):
        value = super().evaluate(epoch, role)
        if epoch == 5:
            value["standard"]["overall"] = {key: 0. for key in value["standard"]["overall"]}
        return value


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.temporary.name) / "fixture"
        seed_all()
        cls.output, cls.journal, cls.old_store = session(cls.fixture)
        cls.adapter = Adapter(B0)
        result = run_session(cls.adapter, PROTOCOL, B0, cls.output, cls.journal, cls.old_store,
            epoch_boundary=lambda epoch, *_: {"status": "CPU_FIXTURE_COMMITTED_EPOCH5"} if epoch == 5 else None)
        assert result["status"] == "CPU_FIXTURE_COMMITTED_EPOCH5"
        cls.old, cls.old_receipt = cls.old_store.load_latest()
        cls.B1 = extension.effective_bindings(B0, ADDENDUM_SHA, NEW_CODE_SHA)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.work = Path(self.temporary.name) / self.id().rsplit(".", 1)[-1]
        self.work.mkdir()

    def cloned_session(self):
        copied = self.work / "session"
        shutil.copytree(self.fixture, copied)
        output = copied / "output"
        journal = RunJournal(copied / "control/updates.jsonl")
        # Completed105 recovery does not consume the old cursor. Legacy-cache
        # identity is tied to these copied fixture paths solely for this test.
        addendum = {"output": str(output.resolve()), "legacy_caches": []}
        for epoch in (3, 4):
            directory = output / "epoch_cache" / f"epoch_{epoch:03d}"
            addendum["legacy_caches"].append({"path": str(directory.resolve()),
                "manifest_sha256": file_sha(directory / "manifest.json"), "epoch": epoch, "bindings": B0})
        addendum["store_directory"] = str((output / "recovery_e10_observation_v1").resolve())
        addendum["best_directory"] = str((output / "best_e10_observation_v1").resolve())
        extension_root = self.work / "extension_control"
        extension_root.mkdir()
        addendum.update(extension_root=str(extension_root.resolve()),
            completion_path=str((extension_root / "completion.json").resolve()),
            epoch10_gate_path=str((extension_root / "epoch010_gate.json").resolve()))
        return output, journal, addendum

    def migrated_store(self, output):
        payload = extension.migrate_payload(self.old, B0, self.B1, ADDENDUM_SHA)
        store = AtomicCheckpointStore(output / "recovery_e10_observation_v1", self.B1)
        receipt = store.save_latest(payload)
        store.save_artifact("derived_update000105_observation.pt", payload, role="epoch_last")
        return payload, store, receipt

    def marker(self, journal, addendum, receipt, *, committed):
        values = {"addendum_sha256": ADDENDUM_SHA, "bindings_sha256": digest(self.B1),
            "store_directory": addendum["store_directory"], "best_directory": addendum["best_directory"],
            "checkpoint_sha256": receipt["sha256"], "origin_checkpoint_sha256": self.old_receipt["sha256"],
            **{key: addendum[key] for key in ("extension_root", "completion_path", "epoch10_gate_path")}}
        event = journal.append("SOURCE_MIGRATION_COMMIT", **values) if committed else {"sha256": "0" * 64}
        return {"schema": extension.SCHEMA + "_ACTIVE_STORE", "addendum_sha256": ADDENDUM_SHA,
            "bindings_sha256": digest(self.B1), "store_directory": addendum["store_directory"],
            "best_directory": addendum["best_directory"], "migration_event_sha256": event["sha256"],
            "migration_checkpoint_sha256": receipt["sha256"],
            **{key: addendum[key] for key in ("extension_root", "completion_path", "epoch10_gate_path")}}

    def test_migration_changes_only_bindings_and_addendum_lineage(self):
        old_sha = file_sha(self.old_store.directory / self.old_receipt["file"])
        before = state_hash(self.old)
        payload = extension.migrate_payload(self.old, B0, self.B1, ADDENDUM_SHA)
        validate_payload(payload, self.B1)
        for field in self.old:
            if field not in ("bindings", "extra"):
                self.assertEqual(state_hash(payload[field]), state_hash(self.old[field]), field)
        self.assertEqual({k: v for k, v in payload["extra"].items() if k != "engineering_observation"}, self.old["extra"])
        self.assertEqual(payload["extra"]["engineering_observation"],
            {"addendum_sha256": ADDENDUM_SHA, "origin_bindings_sha256": digest(B0), "origin_update": 105})
        self.assertEqual(state_hash(self.old), before)
        self.assertEqual(file_sha(self.old_store.directory / self.old_receipt["file"]), old_sha)
        with self.assertRaises(ValueError):
            validate_payload(payload, B0)
        with self.assertRaises(ValueError):
            validate_payload(self.old, self.B1)

    def test_migration_rejects_data_identity_and_extra_attempt(self):
        changed = copy.deepcopy(self.B1)
        changed["manifest_sha256"] = "0" * 64
        with self.assertRaises((HardGate, ValueError)):
            extension.migrate_payload(self.old, B0, changed, ADDENDUM_SHA)
        bad = copy.deepcopy(self.old)
        bad["audit"]["optimizer_attempts"] += 1
        with self.assertRaises((HardGate, ValueError)):
            extension.migrate_payload(bad, B0, self.B1, ADDENDUM_SHA)
        twice = extension.migrate_payload(self.old, B0, self.B1, ADDENDUM_SHA)
        with self.assertRaises((HardGate, ValueError)):
            extension.migrate_payload(twice, self.B1, self.B1, ADDENDUM_SHA)

    def test_boundary_rejects_wrong_epoch_pending_attempt_and_bad_journal(self):
        extension.validate_observation_boundary(self.old, self.journal, 105, 5, B0)
        for update, epoch in ((106, 5), (105, 6), (210, 10)):
            with self.assertRaises((HardGate, ValueError)):
                extension.validate_observation_boundary(self.old, self.journal, update, epoch, B0)
        _, journal, _ = self.cloned_session()
        journal.begin(106)
        with self.assertRaises((HardGate, ValueError)):
            extension.validate_observation_boundary(self.old, journal, 105, 5, B0)
        with self.assertRaises(HardGate):
            RunJournal(journal.path)
        bad_file = self.work / "corrupt_journal.jsonl"
        bad_file.write_bytes(self.journal.path.read_bytes().replace(b'"epoch": 5', b'"epoch": 6', 1))
        with self.assertRaises(HardGate):
            RunJournal(bad_file)

    def test_old_cache_allowlist_and_future_binding_are_exact(self):
        output, _, addendum = self.cloned_session()
        for record in addendum["legacy_caches"]:
            self.assertEqual(extension.legacy_cache_bindings(Path(record["path"]), record["manifest_sha256"], addendum, B0, self.B1), B0)
            with self.assertRaises((HardGate, ValueError)):
                extension.legacy_cache_bindings(Path(record["path"]), "0" * 64, addendum, B0, self.B1)
        future = output / "epoch_cache/epoch_005"
        self.assertEqual(extension.legacy_cache_bindings(future, "f" * 64, addendum, B0, self.B1), self.B1)
        for directory in (output / "epoch_cache/epoch_002", output / "outside/epoch_005"):
            with self.assertRaises((HardGate, ValueError)):
                extension.legacy_cache_bindings(directory, "f" * 64, addendum, B0, self.B1)
        changed = copy.deepcopy(addendum)
        changed["legacy_caches"][0]["bindings"]["manifest_sha256"] = "0" * 64
        with self.assertRaises((HardGate, ValueError)):
            record = changed["legacy_caches"][0]
            extension.legacy_cache_bindings(Path(record["path"]), record["manifest_sha256"], changed, B0, self.B1)

    def test_partial_marker_or_orphan_checkpoint_cannot_be_accepted(self):
        output, journal, addendum = self.cloned_session()
        _, store, receipt = self.migrated_store(output)
        before = file_sha(store.directory / receipt["file"])
        marker = self.marker(journal, addendum, receipt, committed=False)
        with self.assertRaises((HardGate, ValueError)):
            extension.validate_active_store(marker, journal, ADDENDUM_SHA, self.B1, store.directory)
        self.assertEqual(journal.committed, 105)
        self.assertEqual(file_sha(store.directory / receipt["file"]), before)
        valid = self.marker(journal, addendum, receipt, committed=True)
        extension.validate_active_store(valid, journal, ADDENDUM_SHA, self.B1, store.directory)
        for key in ("bindings_sha256", "migration_checkpoint_sha256", "migration_event_sha256", "completion_path", "epoch10_gate_path"):
            altered = {**valid, key: "0" * 64}
            with self.assertRaises((HardGate, ValueError)):
                extension.validate_active_store(altered, journal, ADDENDUM_SHA, self.B1, store.directory)

    def test_bound_file_reference_and_journal_prefix_sha_are_strict(self):
        path = self.work / "CPU_BOUND_EVIDENCE.txt"
        path.write_bytes(b"immutable-prefix\n")
        reference = extension.record(path)
        extension.verify_file(reference)
        with path.open("ab") as stream:
            stream.write(b"permitted-journal-append\n")
        extension.verify_file(reference, prefix=True)
        with self.assertRaises(HardGate):
            extension.verify_file(reference)
        path.write_bytes(b"modified-prefix!\npermitted-journal-append\n")
        with self.assertRaises(HardGate):
            extension.verify_file(reference, prefix=True)

    def test_extended_catalog_rejects_unapproved_source_changes(self):
        repository = self.work / "repository"
        (repository / "tools").mkdir(parents=True)
        (repository / "tools/runtime.py").write_text("VALUE = 1\n")
        relative = "diagnostics/exp17_j2_r5/observation_extension.py"
        path = repository / relative
        path.parent.mkdir(parents=True)
        path.write_text("CPU_EXTENSION_FIXTURE = True\n")
        files = {"tools/runtime.py": file_sha(repository / "tools/runtime.py"), relative: file_sha(path)}
        code = {"files": files, "sha256": digest(files)}
        with patch.object(extension, "__file__", str(path)):
            extension.validate_extended_catalog(repository, code)
            (repository / "tools/runtime.py").write_text("VALUE = 2\n")
            with self.assertRaises((HardGate, ValueError)):
                extension.validate_extended_catalog(repository, code)

    def test_bounded105_to106_and210_to211_preserves_scheduler_and_events(self):
        output, journal, addendum = self.cloned_session()
        payload, store, _ = self.migrated_store(output)
        immutable105 = store.directory / "derived_update000105_observation.pt"
        immutable_sha = file_sha(immutable105)
        old_best = {p.name: file_sha(p) for p in (output / "best").iterdir()}
        old_scheduler = payload["scheduler"]["last_epoch"]
        legacy = lambda directory, sha: extension.legacy_cache_bindings(directory, sha, addendum, B0, self.B1)
        direct = Adapter(B0)
        # CPU optimizer.load_state_dict can alias CPU step tensors. The direct
        # numerical reference must own its copy of the shared immutable fixture.
        restore_training_state(direct.trainer, copy.deepcopy(self.old), bindings=B0)
        direct.start_epoch(5)
        direct.cursor = SimpleNamespace(next_index=1)
        direct.train_batch(synthetic_epoch(5)[0], 106)
        expected106 = capture_training_state(direct.trainer, bindings=B0,
            position={"epoch": 5, "batch_position": 1, "global_update": 106},
            sampler_state={"schema": "CPU_DIRECT_REFERENCE", "epoch": 5, "next_batch": 1})
        adapter = Adapter(self.B1)
        with extension.best_namespace(output, self.B1):
            first = run_session(adapter, PROTOCOL, self.B1, output, journal, store, recovered=payload,
                synthetic_stop_after=106, legacy_cache_bindings=legacy)
            self.assertEqual(first["updates"], 106)
            self.assertEqual(adapter.trainer.scheduler.last_epoch, old_scheduler + 1)
            self.assertEqual(adapter.val_events, [])
            current, _ = store.load_latest()
            for field in ("state", "ema", "optimizer", "scheduler", "scaler", "rng", "audit", "counters", "training_modes"):
                self.assertEqual(state_hash(current[field]), state_hash(expected106[field]), field)
            artifact = output / "CPU_E10_EVIDENCE.json"
            exclusive_json(artifact, {"CPU_SYNTHETIC_ONLY": True})
            accepted = decision(artifact=artifact)
            with patch.object(extension, "check_e10_evidence", return_value=accepted):
                second = run_session(adapter, PROTOCOL, self.B1, output, journal, store, recovered=current,
                    legacy_cache_bindings=legacy, epoch_boundary=lambda e, j, s: extension.epoch_guard(e, j, s,
                        root=output, addendum=addendum, addendum_sha=ADDENDUM_SHA, bindings=self.B1))
            self.assertEqual(second["status"], "R5_EPOCH10_PASS")
            self.assertEqual(journal.committed, 210)
            self.assertEqual(adapter.val_events, [(10, "master"), (10, "ema")])
            self.assertEqual(adapter.diagnostic_epochs, [10])
            saved210, _ = store.load_latest()
            prior_scheduler = saved210["scheduler"]["last_epoch"]
            for name, stopped in (("zero_ap", decision(ap=0.)), ("mean_mismatch", decision(gap=3.))):
                gate_root = self.work / name
                gate_root.mkdir()
                with patch.object(extension, "check_e10_evidence", return_value=stopped):
                    hold = extension.epoch_guard(10, journal, store, root=gate_root, addendum=addendum,
                        addendum_sha=ADDENDUM_SHA, bindings=self.B1)
                self.assertEqual(hold["status"], "R5_EPOCH10_HOLD")
                with self.assertRaises(HardGate):
                    run_session(Adapter(self.B1), PROTOCOL, self.B1, output, journal, store, recovered=saved210,
                        synthetic_stop_after=211, legacy_cache_bindings=legacy,
                        epoch_boundary=lambda e, j, s: extension.epoch_guard(e, j, s, root=gate_root,
                            addendum=addendum, addendum_sha=ADDENDUM_SHA, bindings=self.B1))
                self.assertEqual(journal.committed, 210)
            resumed = Adapter(self.B1)
            third = run_session(resumed, PROTOCOL, self.B1, output, journal, store, recovered=saved210,
                synthetic_stop_after=211, legacy_cache_bindings=legacy,
                epoch_boundary=lambda e, j, s: extension.epoch_guard(e, j, s, root=output,
                    addendum=addendum, addendum_sha=ADDENDUM_SHA, bindings=self.B1))
            self.assertEqual(third["updates"], 211)
            self.assertEqual(resumed.trainer.scheduler.last_epoch, prior_scheduler + 1)
            self.assertEqual(resumed.val_events, [])
        self.assertEqual({p.name: file_sha(p) for p in (output / "best").iterdir()}, old_best)
        self.assertEqual(file_sha(immutable105), immutable_sha)
        archived, _ = store.load_artifact(immutable105.name, expected_sha256=immutable_sha)
        self.assertEqual(archived["position"]["global_update"], 105)
        for field in ("state", "ema", "optimizer", "scheduler", "scaler", "rng", "sampler", "audit", "counters", "training_modes"):
            self.assertEqual(state_hash(archived[field]), state_hash(self.old[field]), field)
        for role in ("master", "ema"):
            receipt = json.loads((output / "best_e10_observation_v1" / f"best_{role}.json").read_text())
            self.assertEqual(receipt["epoch"], 10)
        self.assertFalse(torch.cuda.is_initialized())

    def test_e10_zero_ap_or_mean_mismatch_holds(self):
        self.assertEqual(decision(ap=0.)["status"], "HOLD")
        self.assertEqual(decision(ap=.2, gap=3.)["status"], "HOLD")
        self.assertEqual(decision(ap=.2, gap=2.99)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
