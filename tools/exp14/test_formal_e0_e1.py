import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ultralytics.nn.tasks import SegmentationModel

from tools.exp14.diagnose_amp_first_batch import parameters_sha
from tools.exp14.diagnose_formal_zero_gradients import compare_gradients
from tools.exp14.foreground_audit import count_foreground
from tools.exp14.formal_e0_e1 import (
    INITIAL_STATE_SHA, FormalR1Trainer, FormalTrainer, RevisedAmpSmokeTrainer, hash_state, json_metadata,
    layer_gradient_statistics, validate_checkpoint,
)


class FormalBudgetTests(unittest.TestCase):
    def test_forbids_epoch_101(self):
        with self.assertRaisesRegex(ValueError, "outside|does not match"):
            FormalTrainer.run_callbacks(SimpleNamespace(epoch=100, observed=[]), "on_train_epoch_start")

    def test_epoch_checkpoint_requires_the_exact_budget(self):
        model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=7, verbose=False)
        state = model.state_dict()
        record = {"schema": "exp14_a4_state_v1", "complete": True, "arm": "e0", "epoch": 100,
                  "batches": 2200, "effective_updates": 2200, "skips": 0,
                  "initial_state_sha256": INITIAL_STATE_SHA, "scaler": {"scale": 2, "_growth_tracker": 200},
                  "model_config": model.yaml, "model_state": state, "model_state_sha256": hash_state(state),
                  "parameter_sha256": parameters_sha(model), "optimizer": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(record, path)
            self.assertEqual(validate_checkpoint(path, "e0")["effective_updates"], 2200)
            torch.save({**record, "effective_updates": 2199}, path)
            with self.assertRaisesRegex(ValueError, "budget"):
                validate_checkpoint(path, "e0")

    def test_r1_records_high_zeros_without_stopping_native_updates(self):
        trainer = FormalR1Trainer.__new__(FormalR1Trainer)
        trainer.zero_streak = 50
        with patch.object(RevisedAmpSmokeTrainer, "optimizer_step") as native:
            trainer.optimizer_step()
            native.assert_called_once()
            self.assertIsNone(FormalR1Trainer.zero_streak_limit)
        with patch.object(RevisedAmpSmokeTrainer, "optimizer_step", side_effect=ValueError("native skip")):
            with self.assertRaisesRegex(ValueError, "native skip"):
                trainer.optimizer_step()

    def test_foreground_count_uses_native_scale_column_slices(self):
        mask = torch.tensor([[True, True, False, False, False, True],
                             [False, True, False, False, False, False]])
        self.assertEqual(count_foreground(mask, [2, 3, 1]).tolist(), [3, 0, 1])

    def test_paired_gradient_comparison_quantifies_amp_zero_mass(self):
        result = compare_gradients(torch.tensor([0.0, 3.0]), torch.tensor([4.0, 3.0]))
        self.assertEqual(result["amp_zero_fp32_nonzero"], 1)
        self.assertAlmostEqual(result["amp_zero_fp32_norm_fraction"], 0.8)

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
