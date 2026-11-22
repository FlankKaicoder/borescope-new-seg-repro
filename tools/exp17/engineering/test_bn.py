"""Synthetic CPU tests for native A observation and both no-write N BN modes."""
import unittest

import torch
from torch import nn

from tools.exp17.engineering.bn import (auxiliary_bn, bn_modules, buffer_snapshot,
                                       buffers_equal, capture_native_bn, rng_equal, rng_snapshot)


class BNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(173)

    def test_capture_preserves_native_output_buffers_rng_and_detaches_statistics(self):
        layer = nn.BatchNorm2d(3).train(); reference = nn.BatchNorm2d(3).train()
        reference.load_state_dict(layer.state_dict(), strict=True)
        x = torch.randn(4, 3, 8, 8, requires_grad=True)
        expected = reference(x)
        rng = rng_snapshot()
        with capture_native_bn(layer) as stats: actual = layer(x)
        self.assertTrue(torch.equal(actual, expected)); self.assertTrue(rng_equal(rng))
        self.assertTrue(buffers_equal(layer, buffer_snapshot(reference)))
        torch.testing.assert_close(stats[""][0].mean, x.detach().mean((0, 2, 3)))
        torch.testing.assert_close(stats[""][0].biased_variance, x.detach().var((0, 2, 3), unbiased=False))
        self.assertFalse(stats[""][0].mean.requires_grad)
        self.assertFalse(stats[""][0].biased_variance.requires_grad)

    def test_replay_a_does_not_backprop_through_a_statistics_but_reaches_n_and_affine(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).train()
        a = torch.randn(4, 3, 8, 8, requires_grad=True)
        n = torch.randn(2, 3, 8, 8, requires_grad=True)
        with capture_native_bn(model) as stats: model(a)
        before = buffer_snapshot(model)
        with auxiliary_bn(model, "replay_a", stats): value = model(n)
        value.square().mean().backward()
        self.assertIsNone(a.grad)
        for gradient in (n.grad, model[0].weight.grad, model[1].weight.grad, model[1].bias.grad):
            self.assertTrue(torch.isfinite(gradient).all()); self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertTrue(buffers_equal(model, before))
        self.assertTrue(all("forward" not in m.__dict__ for m in bn_modules(model).values()))

    def test_independent_stats_and_exception_cleanup_never_write_buffers(self):
        model = nn.BatchNorm2d(3).train(); x = torch.randn(2, 3, 8, 8, requires_grad=True)
        before, rng = buffer_snapshot(model), rng_snapshot()
        with auxiliary_bn(model, "independent"): actual = model(x)
        expected = torch.nn.functional.batch_norm(x, None, None, model.weight, model.bias, True, 0, model.eps)
        self.assertTrue(torch.equal(actual, expected)); self.assertTrue(buffers_equal(model, before))
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with auxiliary_bn(model, "independent"):
                model(x); raise RuntimeError("synthetic")
        self.assertNotIn("forward", model.__dict__)
        self.assertTrue(buffers_equal(model, before)); self.assertTrue(rng_equal(rng))

    def test_rng_consumption_is_detected_and_not_concealed(self):
        model = nn.BatchNorm2d(3).train(); before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "RNG"):
            with auxiliary_bn(model, "independent"): torch.rand(1)
        self.assertFalse(torch.equal(before, torch.get_rng_state()))
        self.assertNotIn("forward", model.__dict__)


if __name__ == "__main__": unittest.main()
