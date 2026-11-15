"""Checkpoint and AMP contract tests without forward, backward, or optimizer calls."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.exp14.a11_full_numerics import (
    NumericalFailure, adam_step_counts, expected_scaler_state, read_epoch_checkpoint, save_epoch_checkpoint,
    validate_epoch_metadata, validate_rng_state, validate_scaler_trajectory,
)
from tools.exp14.label_plan import CLASSES


class FormalNumericsTests(unittest.TestCase):
    def test_no_skip_amp_trajectory(self):
        for step, scale, growth_counter in [
            (0, 1., 0), (50, 1., 50), (1999, 1., 1999), (2000, 2., 0), (2001, 2., 1),
            (2100, 2., 100), (3300, 2., 1300),
        ]:
            expected = expected_scaler_state(step)
            self.assertEqual((expected["scale"], expected["_growth_tracker"]), (scale, growth_counter))
            self.assertEqual(validate_scaler_trajectory(expected, step), expected)
        with self.assertRaises(NumericalFailure):
            validate_scaler_trajectory(expected_scaler_state(1999), 2000)
        with self.assertRaises(ValueError):
            expected_scaler_state(3301)

    def test_epoch_and_adam_budget(self):
        metadata = {"target_epochs": 100, "completed_epochs": 2, "epoch_batches": 33, "optimizer_updates": 66}
        self.assertEqual(validate_epoch_metadata("e1", metadata), 66)
        with self.assertRaises(ValueError):
            validate_epoch_metadata("e0", metadata)
        with self.assertRaises(ValueError):
            validate_epoch_metadata("e1", {**metadata, "optimizer_updates": 65})
        self.assertEqual(adam_step_counts({0: {"step": torch.tensor(66.)}, 1: {"step": torch.tensor(10.)}}, 66),
                         {66: 1, 10: 1})
        for wrong in (65., 66.5, float("nan")):
            with self.assertRaises(ValueError):
                adam_step_counts({0: {"step": torch.tensor(wrong)}}, 66)

    def test_checkpoint_is_readable_and_exclusive_with_synthetic_payload(self):
        metadata = {"target_epochs": 100, "completed_epochs": 1, "epoch_batches": 21, "optimizer_updates": 21}
        model = torch.nn.Linear(2, 1)
        model.yaml, model.names = {"name": "fixture"}, dict(enumerate(CLASSES))
        ema = SimpleNamespace(ema=model, updates=21)
        optimizer = SimpleNamespace(
            state_dict=lambda: {"param_groups": [{"params": [0, 1]}],
                                "state": {0: {"step": torch.tensor(21.)}, 1: {"step": torch.tensor(21.)}}},
            param_groups=[{"params": list(model.parameters())}],
        )
        trainer = SimpleNamespace(
            arm="e0", checkpoint_identity={"test": "synthetic fixture"}, model=model, ema=ema, optimizer=optimizer,
            scheduler=SimpleNamespace(state_dict=lambda: {"last_epoch": 0}),
            scaler=SimpleNamespace(state_dict=lambda: expected_scaler_state(21)),
            audit=SimpleNamespace(optimizer_attempts=21, native_steps=21, effective_updates=21, amp_skips=0,
                                  initial_parameters_sha256="fixture_initial_parameters", initial_state_sha256="fixture_initial_state"),
        )
        name, words, position, has_gauss, cached_gauss = np.random.RandomState(7).get_state()
        rng = {"python": random.Random(7).getstate(), "numpy": (name, words.tolist(), position, has_gauss, cached_gauss),
               "torch_cpu": torch.Generator(device="cpu").manual_seed(7).get_state(),
               "torch_cuda": [torch.zeros(1, dtype=torch.uint8)]}
        validate_rng_state(rng)
        with tempfile.TemporaryDirectory() as temp, patch("tools.exp14.a11_full_numerics.capture_rng_state", return_value=rng):
            path = Path(temp) / "epoch_001.pt"
            receipt = save_epoch_checkpoint(trainer, path, metadata)
            self.assertTrue(receipt["cpu_readback_exact"])
            reloaded = read_epoch_checkpoint(path, trainer.checkpoint_identity, "e0")
            self.assertEqual(reloaded["metadata"], metadata)
            self.assertEqual(reloaded["model_state_sha256"], receipt["model_state_sha256"])
            self.assertTrue(torch.equal(reloaded["model_state"]["bias"], model.bias))
            with self.assertRaises(FileExistsError):
                save_epoch_checkpoint(trainer, path, metadata)
            with self.assertRaises(ValueError):
                read_epoch_checkpoint(path, trainer.checkpoint_identity, "e1")
            self.assertEqual(read_epoch_checkpoint(path)["metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
