"""Regression checks for the finite multiseed VAL scope and persistent attempt gate."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.exp15 import je2


class EvaluationScopeTests(unittest.TestCase):
    def test_gate_binds_all_four_states_and_frozen_data(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "preflight.json"
            path.write_text("{}", encoding="utf-8")
            checkpoint_spec = {k: {"file_sha256": f, "native_master_sha256": s}
                               for k, (f, s) in je2.ENDPOINTS.items()}
            gate = je2.gate_record({"identity": {"checkpoint_spec": checkpoint_spec}}, path)
            self.assertEqual(gate["ordered_passes"], list(je2.PASSES))
            self.assertEqual(gate["max_attempts"], 8)
            self.assertEqual(gate["checkpoint_spec"], checkpoint_spec)
            self.assertEqual(gate["protocol_sha256"], je2.PROTOCOL_SHA)
            self.assertEqual(gate["val_files_sha256"], je2.VAL_FILES_SHA)
            changed = {**gate, "ordered_passes": list(reversed(je2.PASSES))}
            self.assertNotEqual(changed, je2.gate_record({"identity": {"checkpoint_spec": checkpoint_spec}}, path))

    def test_attempts_survive_failure_and_cannot_be_reordered_or_repeated(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "claims").mkdir()
            weights = {"seed": 142, "arm": "j0", "epoch": 100}
            je2.reserve_attempt(out, je2.PASSES[0], weights, 0.001)
            with self.assertRaises(ValueError):
                je2.reserve_attempt(out, je2.PASSES[0], weights, 0.001)
            with self.assertRaises(ValueError):
                je2.reserve_attempt(out, je2.PASSES[2], weights, 0.001)
            for name in je2.PASSES[1:]:
                je2.reserve_attempt(out, name, weights, 0.25)
            with self.assertRaises(ValueError):
                je2.reserve_attempt(out, je2.PASSES[-1], weights, 0.25)
            attempts = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines()]
            self.assertEqual([row["ticket"] for row in attempts], list(range(1, 9)))
            self.assertEqual([row["pass"] for row in attempts], list(je2.PASSES))

    def test_wrong_seed_or_arm_is_rejected_before_checkpoint_read(self):
        with patch("torch.load", side_effect=AssertionError("must not read a checkpoint")) as reader:
            for seed, arm in ((42, "j0"), (142, "j2"), (243, "j1")):
                with self.assertRaises(ValueError):
                    je2.read_master(None, seed, arm)
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
