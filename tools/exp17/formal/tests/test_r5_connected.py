"""Independent derivative and early-gate regressions for R5."""
import copy
import unittest

import torch
from torch.nn import functional as F

from tools.exp17.engineering.bn import StatisticsGradient
from tools.exp17.formal.r5_gate import FOCUS, RULE, decide


def normalized(a, n, gamma, beta, *, bridge):
    variance, mean = torch.var_mean(a, (0, 2, 3), unbiased=False)
    if bridge:
        base = F.batch_norm(n, mean.detach(), variance.detach(), gamma, beta, False, 0., .001)
        return StatisticsGradient.apply(base, mean, variance, n.detach(), gamma.detach(), .001)
    return (n - mean[None, :, None, None]) * (variance + .001).rsqrt()[None, :, None, None] * gamma[None, :, None, None] + beta[None, :, None, None]


class ConnectedDerivativeTests(unittest.TestCase):
    def fixture(self, constant=False):
        generator = torch.Generator().manual_seed(17)
        a = torch.randn(3, 2, 3, 2, generator=generator, dtype=torch.float64)
        if constant:
            a = a * 1e-5 + 2
        return [a.requires_grad_(), torch.randn(2, 2, 3, 2, generator=generator, dtype=torch.float64).requires_grad_(),
                torch.tensor([.7, -1.2], dtype=torch.float64, requires_grad=True),
                torch.tensor([.3, -.4], dtype=torch.float64, requires_grad=True)]

    def compare(self, constant=False, layers=1):
        values = self.fixture(constant)
        def execute(bridge):
            a, n, gamma, beta = values
            y = normalized(a, n, gamma, beta, bridge=bridge)
            if layers == 2:
                y = normalized(a.sin(), y.sin(), gamma, beta, bridge=bridge)
            loss = (y.sin() + y.square() * .03).sum()
            return y, torch.autograd.grad(loss, values, retain_graph=True)
        native, actual = execute(True)
        reference, expected = execute(False)
        torch.testing.assert_close(native, reference, atol=1e-11, rtol=1e-11)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-9)

    def test_first_order_reference(self):
        self.compare()

    def test_low_variance_reference(self):
        self.compare(constant=True)

    def test_multilayer_chain_rule(self):
        self.compare(layers=2)

    def test_independent_central_differences(self):
        values = self.fixture()
        loss = normalized(*values, bridge=True).sin().sum()
        gradients = torch.autograd.grad(loss, values)
        with torch.no_grad():
            for tensor, gradient in zip(values, gradients):
                for index in (0, tensor.numel() - 1):
                    flat = tensor.view(-1)
                    old = flat[index].item()
                    flat[index] = old + 1e-6
                    plus = normalized(*values, bridge=False).sin().sum().item()
                    flat[index] = old - 1e-6
                    minus = normalized(*values, bridge=False).sin().sum().item()
                    flat[index] = old
                    self.assertAlmostEqual(gradient.view(-1)[index].item(), (plus - minus) / 2e-6, places=6)

    def test_bitwise_forward_and_inplace_silu(self):
        a, n, gamma, beta = self.fixture()
        variance, mean = torch.var_mean(a, (0, 2, 3), unbiased=False)
        base = F.batch_norm(n, mean.detach(), variance.detach(), gamma, beta, False, 0., .001)
        result = StatisticsGradient.apply(base, mean, variance, n.detach(), gamma.detach(), .001)
        self.assertTrue(torch.equal(base, result))
        F.silu(result, inplace=True).sum().backward()
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in (a, n, gamma, beta)))

    def test_repeated_component_backward(self):
        values = self.fixture()
        y = normalized(*values, bridge=True)
        first = torch.autograd.grad(y.square().sum(), values, retain_graph=True)
        second = torch.autograd.grad(y.square().sum(), values, retain_graph=True)
        self.assertTrue(all(torch.equal(x, z) for x, z in zip(first, second)))
        self.assertTrue(all(v.grad is None for v in values))


class EarlyGateTests(unittest.TestCase):
    def fixture(self):
        reports = [{"batch_position": p, "bn_mean_gap": {n: {"D": 1.} for n in FOCUS}} for p in RULE["positions"]]
        metrics = {"standard": {"overall": {"metrics/mAP50-95(M)": .01}}}
        return reports, metrics

    def test_positive_ap_matching_statistics(self):
        self.assertEqual(decide(*self.fixture())["status"], "PASS")

    def test_zero_ap_hold(self):
        reports, metrics = self.fixture()
        metrics["standard"]["overall"]["metrics/mAP50-95(M)"] = 0
        self.assertEqual(decide(reports, metrics)["status"], "HOLD")

    def test_threshold_is_inclusive_three_of_four(self):
        reports, metrics = self.fixture()
        for r in reports[:3]:
            r["bn_mean_gap"][FOCUS[-1]]["D"] = 3.
        self.assertEqual(decide(reports, metrics)["status"], "HOLD")
        reports[0]["bn_mean_gap"][FOCUS[-1]]["D"] = 2.9999
        self.assertEqual(decide(reports, metrics)["status"], "PASS")

    def test_nonfinite_or_missing_probe_rejected(self):
        reports, metrics = self.fixture()
        reports[0]["bn_mean_gap"][FOCUS[0]]["D"] = float("nan")
        with self.assertRaises(ValueError):
            decide(reports, metrics)
        with self.assertRaises(ValueError):
            decide(reports[1:], metrics)


if __name__ == "__main__":
    unittest.main()
