import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.exp14.formal_e0_e1 import (
    INITIAL_STATE_SHA, FormalTrainer, hash_state, json_metadata, layer_gradient_statistics, validate_checkpoint,
)


class FormalBudgetTests(unittest.TestCase):
    def test_forbids_epoch_101(self):
        with self.assertRaisesRegex(ValueError, "outside|does not match"):
            FormalTrainer.run_callbacks(SimpleNamespace(epoch=100, observed=[]), "on_train_epoch_start")

    def test_epoch_checkpoint_requires_the_exact_budget(self):
        state = {"weight": torch.tensor([1.0])}
        record = {"schema": "exp14_a4_state_v1", "complete": True, "arm": "e0", "epoch": 100,
                  "batches": 2200, "effective_updates": 2200, "skips": 0,
                  "initial_state_sha256": INITIAL_STATE_SHA, "scaler": {"scale": 2, "_growth_tracker": 200},
                  "model_state": state, "model_state_sha256": hash_state(state), "optimizer": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(record, path)
            self.assertEqual(validate_checkpoint(path, "e0")["effective_updates"], 2200)
            torch.save({**record, "effective_updates": 2199}, path)
            with self.assertRaisesRegex(ValueError, "budget"):
                validate_checkpoint(path, "e0")

    def test_checkpoint_metadata_normalizes_torch_version_to_safe_string(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.pt"
            torch.save(json_metadata({"torch": torch.__version__}), path)
            self.assertEqual(torch.serialization.get_unsafe_globals_in_checkpoint(path), [])
            metadata = torch.load(path, map_location="cpu", weights_only=True)
            self.assertIs(type(metadata["torch"]), str)
            self.assertEqual(metadata["torch"], str(torch.__version__))

    def test_zero_and_nonfinite_population_by_module(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
        model[0].weight.grad = torch.tensor([[0.0, float("inf")]])
        stats = layer_gradient_statistics(model)["0.weight"]
        self.assertEqual((stats["elements"], stats["zero"], stats["nonfinite"]), (2, 1, 1))
        self.assertEqual(stats["finite_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
