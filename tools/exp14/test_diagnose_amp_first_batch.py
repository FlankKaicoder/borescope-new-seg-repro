import unittest

import torch

from tools.exp14.diagnose_amp_first_batch import (
    NeverUpdate, classify, gradient_stats, parameters_sha, scaler_probe, tensor_digest,
)


class NumericalDiagnosisTests(unittest.TestCase):
    def test_nonfinite_gradient_statistics_preserve_finite_subset(self):
        result = gradient_stats(torch.tensor([[float("inf"), float("-inf"), float("nan"), -2.0, 4.0]]))
        self.assertEqual(result["shape"], [1, 5])
        self.assertEqual((result["nonfinite"], result["nan"], result["posinf"], result["neginf"]), (3, 1, 1, 1))
        self.assertEqual((result["finite_min"], result["finite_max"], result["finite_abs_mean"]), (-2.0, 4.0, 3.0))

    def test_native_scaler_can_backoff_without_a_parameter_update(self):
        model = torch.nn.Linear(2, 1, bias=False)
        before = parameters_sha(model)
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        scaler = torch.amp.GradScaler("cpu", init_scale=4096)
        scaler.scale(torch.tensor(1.0))
        unscaled, probe = scaler_probe(model, scaler, check_skip=True)
        self.assertFalse(unscaled["all_finite"])
        self.assertEqual(probe["found_inf"], 1)
        self.assertTrue(probe["native_skip_probed"])
        self.assertEqual(probe["guard_step_calls"], 0)
        self.assertEqual(scaler.get_scale(), 2048)
        self.assertEqual(parameters_sha(model), before)

    def test_guard_has_no_implementation_that_can_update_a_parameter(self):
        value = torch.nn.Parameter(torch.tensor([3.0]))
        guard = NeverUpdate([value])
        with self.assertRaisesRegex(AssertionError, "Diagnostic optimizer"):
            guard.step()
        self.assertEqual(value.item(), 3.0)

    def test_tensor_digest_is_value_sensitive(self):
        value = torch.tensor(3.0)
        self.assertEqual(tensor_digest(value), tensor_digest(value.clone()))
        self.assertNotEqual(tensor_digest(value), tensor_digest(value + 1))

    def test_classifies_only_complete_five_case_matrix(self):
        def rows(results):
            return [{"status": "COMPLETE", "scaled_or_fp32_gradients": {"all_finite": r}} for r in results]
        checks = (
            ([False, True, True, True, True], "AMP_SCALE_OVERFLOW_LIKELY"),
            ([False, False, False, False, True], "AMP_BACKWARD_NUMERICAL_ISSUE"),
            ([False, False, False, False, False], "FP32_BACKWARD_NONFINITE"),
            ([True, True, True, True, True], "NONREPRODUCIBLE"),
            ([False, True], "INSUFFICIENT_EVIDENCE"),
        )
        for values, expected in checks:
            with self.subTest(values=values):
                self.assertEqual(classify(rows(values)), expected)


if __name__ == "__main__":
    unittest.main()
