import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.exp14 import a11_full_protocol as protocol


class FullProtocolTest(unittest.TestCase):
    def test_primary_pair_preserves_same_original_exposure_not_same_updates(self):
        spec = protocol.protocol_spec()
        e0, e1 = spec["arms"].values()
        self.assertEqual((e0["effective_updates"], e1["effective_updates"]), (2100, 3300))
        self.assertEqual((e0["warmup_iterations"], e1["warmup_iterations"]), (63, 99))
        self.assertEqual((e0["tail_batch"], e1["tail_batch"]), (28, 23))
        self.assertEqual(e0["original_exposures"], e1["original_exposures"], 66800)
        self.assertEqual(e0["raw_instance_exposures"], e1["raw_instance_exposures"], 126600)
        self.assertEqual(e1["repair_exposures"], 37900)
        self.assertFalse(spec["evaluation"]["test_model_inference"])

    def test_loading_rejects_changed_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            spec = protocol.protocol_spec()
            path.write_text(json.dumps(spec))
            self.assertEqual(protocol.load_protocol(path), spec)
            spec["arms"]["e0"]["effective_updates"] = 3300
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "preregistered contract"):
                protocol.load_protocol(path)

    def test_distinct_gate_requires_both_matching_file_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config, identity, gate_path = (path / name for name in ("p.json", "i.json", "a.json"))
            config.write_text(json.dumps(protocol.protocol_spec()))
            identity.write_text("identity")
            gate = {"stage": "EXP14_E0_E1_FULL_TRAIN_AUTHORIZATION", "decision": "PASS",
                    "protocol_sha256": protocol.protocol_sha256(config),
                    "identity_sha256": protocol.protocol_sha256(identity), "authorized_arms": ["e0", "e1"]}
            with patch.object(protocol, "validate_identity") as validate:
                gate_path.write_text(json.dumps(gate))
                self.assertEqual(protocol.authorize_training(gate_path, config, identity, "e0"), gate)
                validate.assert_called_once_with(identity, config)
                identity.write_text("different")
                with self.assertRaisesRegex(ValueError, "authorization"):
                    protocol.authorize_training(gate_path, config, identity, "e1")
                with self.assertRaisesRegex(ValueError, "primary arms"):
                    protocol.authorize_training(gate_path, config, identity, "e0-update")
