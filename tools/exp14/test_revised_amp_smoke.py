import unittest

import torch

from tools.exp14.revised_amp_smoke import (
    NATIVE_DEFAULTS, active_groups, clip_native_gradients, gradient_population,
    parameter_vector, unscale_and_inspect,
)


class RevisedAmpTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
        self.optimizer = torch.optim.AdamW([{"params": [self.weight], "param_group": "weight"}], lr=0.01)
        self.scaler = torch.amp.GradScaler("cpu", init_scale=1)
        self.calls = []
        self.optimizer.register_step_pre_hook(lambda *args: self.calls.append("step"))

    def test_finite_native_step_changes_parameters_and_keeps_scale(self):
        self.scaler.scale(self.weight.square().sum()).backward()
        scaled, unscaled, overflow = unscale_and_inspect(self.scaler, self.optimizer, [self.weight])
        self.assertFalse(overflow)
        self.assertEqual(scaled, unscaled)
        self.assertEqual(unscaled, {"tensors": 1, "elements": 2, "nonfinite": 0, "zero": 0})
        self.assertGreater(clip_native_gradients([self.weight]), 0)
        before = parameter_vector([self.weight])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.assertEqual(self.calls, ["step"])
        self.assertTrue(bool((self.weight != before).any()))
        self.assertEqual(self.scaler.get_scale(), 1)
        self.assertEqual({k: self.scaler.state_dict()[k] for k in NATIVE_DEFAULTS}, NATIVE_DEFAULTS)

    def test_overflow_does_not_call_adamw_and_backs_off(self):
        self.scaler.scale(self.weight.square().sum()).backward()
        self.weight.grad[0] = float("inf")
        before = parameter_vector([self.weight])
        scaled, unscaled, overflow = unscale_and_inspect(self.scaler, self.optimizer, [self.weight])
        self.assertTrue(overflow)
        self.assertEqual((scaled["nonfinite"], unscaled["nonfinite"]), (1, 1))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.assertEqual(self.calls, [])
        self.assertTrue(torch.equal(self.weight, before))
        self.assertEqual(self.scaler.get_scale(), 0.5)

    def test_native_growth_boundary_is_after_two_thousand_finite_steps(self):
        state = self.scaler.state_dict()
        self.scaler.load_state_dict({**state, "_growth_tracker": state["growth_interval"] - 1})
        self.scaler.scale(self.weight.square().sum()).backward()
        _, _, overflow = unscale_and_inspect(self.scaler, self.optimizer, [self.weight])
        self.assertFalse(overflow)
        self.assertEqual(self.scaler.get_scale(), 1)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.assertEqual(self.calls, ["step"])
        self.assertEqual(self.scaler.get_scale(), 2)

    def test_population_separates_zero_and_nonfinite(self):
        self.weight.grad = torch.tensor([0.0, float("nan")])
        self.assertEqual(gradient_population([self.weight]),
                         {"tensors": 1, "elements": 2, "nonfinite": 1, "zero": 1})

    def test_zero_lr_groups_are_excluded_from_parameter_change_evidence(self):
        self.assertEqual(active_groups(self.optimizer), [("weight", [self.weight])])
        self.optimizer.param_groups[0]["lr"] = 0
        self.assertEqual(active_groups(self.optimizer), [])


if __name__ == "__main__":
    unittest.main()
