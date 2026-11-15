import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.exp14.a11_full_train import (
    APPROVED_RESUME, authorize_resume, check_epoch_counts, discarded_ledger, epoch_seed, restore_ledger, run,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalTrainSafetyTests(unittest.TestCase):
    def test_epoch_seed_is_stable_and_distinct(self):
        schedule = [epoch_seed(42, epoch) for epoch in range(100)]
        self.assertEqual(len(set(schedule)), 100)
        self.assertEqual(schedule[12], epoch_seed(42, 12))
        self.assertNotEqual(schedule[12], epoch_seed(43, 12))
        with self.assertRaises(ValueError):
            epoch_seed(42, 100)

    def test_coverage_requires_full_original_and_repair_sets_and_tail(self):
        counts = {"original_exposures": 2, "repair_exposures": 1, "raw_instance_exposures": 4}
        spec = {"batches_per_epoch": 2, "tail_batch": 1, "original_images": 2, "repair_images": 1, "raw_instances": 4}
        expected = {"original-a", "original-b", "repair-a"}
        check_epoch_counts(counts, spec, 2, 1, expected, set(expected))
        with self.assertRaisesRegex(ValueError, "membership"):
            check_epoch_counts(counts, spec, 2, 1, expected, {"original-a", "repair-a"})
        with self.assertRaisesRegex(ValueError, "exposure"):
            check_epoch_counts({**counts, "raw_instance_exposures": 3}, spec, 2, 1, expected, set(expected))

    def test_prep_verifies_without_importing_or_building_the_trainer(self):
        with tempfile.TemporaryDirectory() as docs:
            root = Path(docs)
            for name in ("protocol", "identity", "weight", "manifest"):
                (root / name).write_bytes(b"verified test input")
            release = root / "frozen"
            release.mkdir()
            args = SimpleNamespace(output=root / "prep", release=release, protocol=root / "protocol",
                                   identity=root / "identity", weight=root / "weight", deployment_manifest=root / "manifest",
                                   arm="e0", apply_gate=None, resume_checkpoint=None, resume_gate=None)
            with patch("tools.exp14.a11_full_train.load_protocol", return_value={}), \
                    patch("tools.exp14.a11_full_train.validate_identity"), \
                    patch("tools.exp14.a11_full_train.verify_inputs", return_value={"data": "verified"}), \
                    patch("tools.exp14.a11_full_train.build_formal_trainer") as not_built:
                report = run(args)
            not_built.assert_not_called()
            self.assertEqual(report["status"], "PREP_ONLY")
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertEqual(report["data_before"], {"data": "verified"})

    def test_resume_requires_pinned_latest_checkpoint_and_clean_prior_event_ledger(self):
        with tempfile.TemporaryDirectory() as docs:
            root = Path(docs)
            prior = root / "prior"
            checkpoints = prior / "checkpoints"
            checkpoints.mkdir(parents=True)
            cp = checkpoints / "epoch_005.pt"
            cp.write_bytes(b"test checkpoint")
            events = prior / "events.jsonl"
            events.write_bytes(b'{"event":"epoch_complete","epoch":5}\n')
            (prior / "process.json").write_text(json.dumps({"host": "non-local-test", "pid": 123,
                                                          "linux_process_start": "test"}), encoding="utf-8")
            training_gate = root / "training_gate.json"
            training_gate.write_bytes(b"train gate")
            resume_gate = root / "resume_gate.json"
            identities = {"protocol_sha256": "protocol", "identity_sha256": "identity"}
            data = {**identities, "stage": APPROVED_RESUME, "decision": "PASS", "arm": "e0",
                    "training_authorization_sha256": digest(training_gate), "checkpoint_sha256": digest(cp),
                    "prior_events_sha256": digest(events)}
            resume_gate.write_text(json.dumps(data), encoding="utf-8")
            args = SimpleNamespace(resume_checkpoint=cp, resume_gate=resume_gate, output=root / "next",
                                   apply_gate=training_gate, arm="e0")
            self.assertEqual(authorize_resume(args, identities)[0], cp)
            (checkpoints / "epoch_010.pt").write_bytes(b"newer checkpoint")
            with self.assertRaisesRegex(ValueError, "latest sealed"):
                authorize_resume(args, identities)
            resumed = checkpoints / "epoch_010.pt"
            events.write_bytes(events.read_bytes() + b'{"event":"optimizer_stopped","batch":222}\n')
            args.resume_checkpoint = resumed
            data.update(checkpoint_sha256=digest(resumed), prior_events_sha256=digest(events))
            resume_gate.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "numerical"):
                authorize_resume(args, identities)

    def test_restore_pins_previous_complete_epoch_prefix(self):
        with tempfile.TemporaryDirectory() as docs:
            event_file = Path(docs) / "events.jsonl"
            completed_prefix = b'{"event":"epoch_complete","epoch":5}\n'
            event_file.write_bytes(completed_prefix + b'{"event":"batch_read","batch":106}\n')
            counts = dict.fromkeys(("batches_read", "forwards", "backwards", "finite_forwards", "completed_backwards"), 105)
            counts.update(complete_epochs=5, original_exposures=3340, repair_exposures=0, raw_instance_exposures=6330)
            metadata = {"completed_epochs": 5, "target_epochs": 100, "optimizer_updates": 105, "epoch_batches": 21,
                        "epoch_ledger_sha256": hashlib.sha256(completed_prefix).hexdigest(), "event_bytes": len(completed_prefix),
                        "counters": counts}
            spec = {"batches_per_epoch": 21, "original_images": 668, "repair_images": 0, "raw_instances": 1266}
            self.assertEqual(restore_ledger(metadata, event_file, spec)[0], 5)
            event_file.write_bytes(b'{"event":"epoch_complete","epoch":4}\n')
            with self.assertRaisesRegex(ValueError, "prefix"):
                restore_ledger(metadata, event_file, spec)

    def test_uncommitted_epoch_work_is_counted_separately(self):
        with tempfile.TemporaryDirectory() as docs:
            event_file = Path(docs) / "events.jsonl"
            prefix = b'{"event":"epoch_complete","epoch":5}\n'
            event_file.write_bytes(prefix + b'{"event":"batch_read","original_images":7,"repaired_images":1}\n'
                                   b'{"event":"optimizer","native_step_called":true}\n'
                                   b'{"event":"batch_read","original_images":6,"repaired_images":2}\n'
                                   b'{"event":"forwa')
            report = discarded_ledger(event_file, len(prefix))
            self.assertEqual(report["known_discarded_batches"], 2)
            self.assertEqual(report["known_discarded_native_steps"], 1)
            self.assertEqual(report["known_discarded_original_exposures"], 13)
            self.assertEqual(report["known_discarded_repair_exposures"], 3)
            self.assertEqual(report["potentially_inflight_batches"], 1)
            self.assertTrue(report["trailing_partial_event"])
