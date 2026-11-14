"""CPU-only synthetic checks for A11 bounded numerical stopping and native AMP."""

import unittest
from types import SimpleNamespace

import torch

from tools.exp14.a11_smoke_numerics import NumericalAudit, NumericalFailure, exact_tree, finite_tree


class NumericalAuditTests(unittest.TestCase):
    def audit(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=True), torch.nn.Linear(2, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
        audit = NumericalAudit(SimpleNamespace(model=model, optimizer=optimizer, scaler=scaler))
        self.addCleanup(audit.close)
        return audit, model

    def test_native_step_changes_master_parameters_even_with_structured_zeros(self):
        audit, model = self.audit()
        loss = model(torch.zeros(2, 3)).square().sum() + model[-1].bias.sum()
        self.assertEqual(audit.check_forward(loss, {"cls_loss": torch.ones(())})["loss_items"], {"cls_loss": 1.0})
        audit.scaler.scale(loss).backward()
        result = audit.optimizer_step()
        self.assertTrue(result["parameter_changed"])
        self.assertTrue(result["optimizer_state_finite"])
        self.assertGreater(result["unscaled_gradients"]["zero"], 0)
        self.assertIn("layer_gradients", result)
        self.assertEqual(audit.optimizer_attempts, audit.native_steps, audit.effective_updates)
        self.assertEqual((audit.effective_updates, audit.amp_skips, result["scale_after"]), (1, 0, 1))

    def test_scaled_overflow_cannot_be_counted_as_an_update(self):
        audit, model = self.audit()
        model[0].weight.grad = torch.full_like(model[0].weight, float("inf"))
        audit.scaler.scale(torch.ones(()))
        with self.assertRaises(NumericalFailure) as raised:
            audit.optimizer_step()
        report = raised.exception.evidence
        self.assertTrue(report["native_overflow_handling_as_expected"])
        self.assertEqual((report["scale_before"], report["scale_after"]), (1, 0.5))
        self.assertEqual((audit.optimizer_attempts, audit.amp_skips, audit.native_steps, audit.effective_updates),
                         (1, 1, 0, 0))

    def test_global_zero_and_nonfinite_loss_stop_before_native_step(self):
        audit, model = self.audit()
        loss = model[0].weight.sum() * 0
        with self.assertRaises(NumericalFailure) as raised:
            audit.check_forward(torch.tensor(float("nan")), torch.tensor([1.0, float("inf")]))
        self.assertEqual(raised.exception.evidence, {"loss": None, "loss_items": [1.0, None]})
        audit.scaler.scale(loss).backward()
        with self.assertRaises(NumericalFailure) as raised:
            audit.optimizer_step()
        self.assertFalse(raised.exception.evidence["optimizer_attempted"])
        self.assertEqual(audit.native_steps, 0)

    def test_diagnostic_backward_clips_without_stepping_or_growing(self):
        audit, model = self.audit()
        original = model[0].weight.detach().clone()
        audit.scaler.scale(model[0].weight.square().sum()).backward()
        report = audit.diagnostic_backward()
        self.assertGreater(report["gradient_norm"], 0)
        self.assertTrue(torch.equal(original, model[0].weight))
        self.assertEqual(audit.scaler.state_dict()["_growth_tracker"], 0)
        self.assertEqual((audit.optimizer_attempts, audit.native_steps, audit.amp_skips, audit.effective_updates), (0, 0, 0, 0))

    def test_state_check_preserves_tensor_dtype_and_detects_nonfinite(self):
        self.assertFalse(finite_tree({"adam": [torch.ones(2), torch.tensor([float("inf")])]}))
        self.assertFalse(finite_tree({"lr": float("nan")}))
        self.assertFalse(exact_tree({"steps": torch.tensor([1, 2])}, {"steps": torch.tensor([1., 2.])}))
        self.assertTrue(exact_tree({"steps": torch.tensor([1, 2])}, {"steps": torch.tensor([1, 2])}))


if __name__ == "__main__":
    unittest.main()
