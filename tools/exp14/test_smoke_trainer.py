import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.exp14.freeze_core import FrozenSlotBatchSampler
from tools.exp14.smoke_trainer import SmokeTrainer, TaggedBatchSampler, finite, verify_payload


class SmokeSafetyTests(unittest.TestCase):
    def test_transport_binds_exact_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sample = root / "training.txt"
            sample.write_bytes(b"synthetic")
            manifest = {"scope": "TRAIN_ONLY", "test_files": 0, "files": [
                {"path": "training.txt", "bytes": 9, "sha256": hashlib.sha256(b"synthetic").hexdigest()}
            ]}
            (root / "transport_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_payload(root), manifest)
            sample.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_payload(root)
            sample.write_bytes(b"synthetic")
            (root / "unexpected.txt").write_bytes(b"synthetic")
            with self.assertRaisesRegex(ValueError, "file set"):
                verify_payload(root)

    def test_tagged_sampler_keeps_e0_duplicate_identity_and_epoch_order(self):
        slots = [{"slot_id": "base:x", "e0_image_path": "x.jpg"},
                 {"slot_id": "extra:x", "e0_image_path": "x.jpg"},
                 {"slot_id": "base:y", "e0_image_path": "y.jpg"}]
        schedule = [{"epoch": 1, "batches": [["base:y", "extra:x"], ["base:x"]]},
                    {"epoch": 2, "batches": [["base:x", "base:y"], ["extra:x"]]}]
        sampler = FrozenSlotBatchSampler(slots, schedule, ["x.jpg", "y.jpg"], "e0")
        tagged = TaggedBatchSampler(sampler)
        self.assertEqual(list(tagged), [[(1, "base:y"), (0, "extra:x")], [(0, "base:x")]])
        sampler.set_epoch(2)
        self.assertEqual(list(tagged), [[(0, "base:x"), (1, "base:y")], [(0, "extra:x")]])

    def test_third_epoch_and_unexpected_val_path_raise(self):
        with self.assertRaisesRegex(ValueError, "beyond two epochs"):
            SmokeTrainer.run_callbacks(SimpleNamespace(epoch=2), "on_train_epoch_start")
        with self.assertRaisesRegex(ValueError, "Evaluation path"):
            SmokeTrainer.get_dataloader(SimpleNamespace(), "synthetic", mode="val")
        self.assertIsNone(SmokeTrainer.get_dataloader(SimpleNamespace(), None, mode="val"))

    def test_nonfinite_tensor_is_detected_without_model(self):
        self.assertTrue(finite({"loss": torch.tensor([0.0, 1.0])}))
        self.assertFalse(finite({"loss": [torch.tensor([float("inf")])]}))
        self.assertFalse(finite({"loss": torch.tensor([float("nan")])}))


if __name__ == "__main__":
    unittest.main()
