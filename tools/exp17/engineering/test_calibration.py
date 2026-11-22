"""Synthetic proofs of algebra, state isolation and frozen master selection."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.exp14.formal_e0_e1 import hash_state
from tools.exp17.engineering.bn import buffer_snapshot, buffers_equal, rng_equal, rng_snapshot
from tools.exp17.engineering.calibration import (candidate_algebra, compute_probe, frozen_state,
                                                isolated_model, load_reference, reject_optimizer_step, sha,
                                                verify_checkpoint_benchmark)
from tools.exp17.engineering.calibration_rules import LAMBDA_CANDIDATES
from tools.exp17.engineering.model import native_state_dict
from tools.exp17.engineering import test_model as model_fixtures


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): model_fixtures.ModelTests.setUpClass()

    def test_seven_candidates_are_exact_joint_gradient_and_global_clip_algebra(self):
        y = torch.tensor([3., 0., 0.], dtype=torch.float64)
        s = torch.tensor([0., 4., 0.], dtype=torch.float64)
        p = torch.tensor([0., 0., 2.], dtype=torch.float64)
        rows = candidate_algebra(y, s, p, {"backbone": torch.arange(3)})
        self.assertEqual([r["lambda_pair"] for r in rows], list(LAMBDA_CANDIDATES))
        for row in rows:
            self.assertAlmostEqual(row["joint_all_trainable_norm"], (25 + (2 * row["lambda_pair"]) ** 2) ** .5)
            self.assertAlmostEqual(row["weighted_pair_to_main_joint_backbone_ratio"], 2 * row["lambda_pair"] / 5)

    def test_isolation_restores_bn_model_rng_and_existing_gradient_buffers_even_on_exception(self):
        model = copy.deepcopy(model_fixtures.ModelTests.base); before = hash_state(model.state_dict()); rng = rng_snapshot()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with isolated_model(model):
                model.model[0].bn.running_mean.add_(1); torch.rand(1); raise RuntimeError("synthetic")
        self.assertEqual(hash_state(model.state_dict()), before); self.assertTrue(rng_equal(rng))

    def test_reference_requires_fp32_master_exact_file_native_hash_and_registered_updates(self):
        model = copy.deepcopy(model_fixtures.ModelTests.base); state = frozen_state(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            torch.save({"arm": "j1", "effective_updates": 105, "state": state,
                        "model_state_sha256": hash_state(state), "ema_state_sha256": "different_ema",
                        "native_state_sha256": hash_state(native_state_dict(model)), "scaler": {"scale": 1.0}}, path)
            spec = {"path": str(path), "sha256": sha(path), "step": 105,
                    "expected_native_state_sha256": hash_state(native_state_dict(model))}
            self.assertTrue(load_reference(model, spec, "j1_epoch005")["master_selected_not_ema"])
            with self.assertRaises(ValueError): load_reference(model, {**spec, "sha256": "tamper"}, "j1_epoch005")
            with self.assertRaises(ValueError): load_reference(model, spec, "j1_epoch050")

    def test_probe_restores_state_and_exposes_registered_bn_and_perclass_gradient_evidence(self):
        model = copy.deepcopy(model_fixtures.ModelTests.base); batch, ar, n, nm, nr, bindings = model_fixtures.synthetic()
        prepared = SimpleNamespace(batch=batch, a_regions=ar, normal_images=n, normal_masks=nm,
                                   normal_regions=nr, bindings=bindings)
        before, bn, rng = hash_state(model.state_dict()), buffer_snapshot(model), rng_snapshot()
        result = compute_probe(model, prepared, state="coco_initial", batch_position=1, checkpoint_prefix=True)
        self.assertEqual(result["model_state_sha256_before"], result["model_state_sha256_after"])
        self.assertEqual(before, hash_state(model.state_dict())); self.assertTrue(buffers_equal(model, bn)); self.assertTrue(rng_equal(rng))
        self.assertEqual(set(result["bn_modes"]), {"independent", "replay_a"})
        self.assertEqual(set(result["global_pair_class_component_norms"]), {"0", "1"})
        self.assertTrue(result["bn_modes"]["replay_a"]["N_equals_A_invariance"]["pass"])
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(len(result["lambda_candidates_algebra"]), 7)

    def test_optimizer_guard_rejects_before_a_real_adamw_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0])); optimizer = torch.optim.AdamW([parameter], lr=.1)
        parameter.grad = torch.ones_like(parameter); before = parameter.detach().clone()
        handle = optimizer.register_step_pre_hook(reject_optimizer_step)
        try:
            with self.assertRaisesRegex(ValueError, "optimizer.step"):
                optimizer.step()
            self.assertTrue(torch.equal(parameter, before)); self.assertEqual(optimizer.state, {})
        finally: handle.remove()

    def test_checkpoint_optimization_receipt_binds_exact_model_bn_loss_sources(self):
        directory = Path(__file__).parent
        report = {"status": "PASS_PREFIX_EQUIVALENCE_AND_SPARSE_COUNTERFACTUAL", "checkpoint_prefix_permitted": True,
                  "optimizer_updates": 0, "identity": {"manifest_sha256": "SYNTHETIC_MANIFEST", "batch_sha256": "SYNTHETIC_BATCH",
                  "rules_sha256": sha(directory / "calibration_rules.py"),
                  "model_sources_sha256": {name: sha(directory / name) for name in ("model.py", "bn.py", "losses.py")}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_benchmark.json"; path.write_text(json.dumps(report), encoding="utf-8")
            config = {"checkpoint_prefix_benchmark_approved": True, "manifest_sha256": "SYNTHETIC_MANIFEST",
                      "checkpoint_prefix_benchmark_receipt": {"path": str(path), "sha256": sha(path)}}
            fixed = [{"batch_position": 1, "sha256": "SYNTHETIC_BATCH"}]
            self.assertEqual(verify_checkpoint_benchmark(config, fixed), sha(path))
            report["identity"]["model_sources_sha256"]["bn.py"] = "TAMPERED"
            path.write_text(json.dumps(report), encoding="utf-8"); config["checkpoint_prefix_benchmark_receipt"]["sha256"] = sha(path)
            with self.assertRaisesRegex(ValueError, "source changed"): verify_checkpoint_benchmark(config, fixed)


if __name__ == "__main__": unittest.main()
