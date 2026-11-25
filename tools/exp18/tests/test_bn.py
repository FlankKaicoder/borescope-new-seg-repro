import unittest
from unittest.mock import patch

import torch
from torch import nn

from tools.exp18.bn import (auxiliary_bn, buffer_snapshot, buffers_equal,
                           capture_native_bn, cpu_rng_state, rng_equal,
                           model_rng_state, rng_model_equal)


class BatchNormTests(unittest.TestCase):
    def test_uninitialized_cuda_rng_getters_are_never_called(self):
        with patch("torch.cuda.is_initialized", return_value=False), \
             patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("CUDA RNG getter called")) as getter, \
             patch("torch.cuda.get_rng_state", side_effect=AssertionError("CUDA RNG getter called")), \
             patch("torch.cuda.is_available", side_effect=AssertionError("CUDA availability queried")):
            before = model_rng_state()
            self.assertTrue(rng_model_equal(before, model_rng_state()))
            self.assertIsNone(before["cuda"])
            getter.assert_not_called()

    def test_auxiliary_cpu_stochastic_operation_is_detected_without_rng_restore(self):
        bn = nn.BatchNorm2d(2).train()
        a = torch.randn(2, 2, 2, 2, requires_grad=True)
        normal = torch.randn(2, 2, 2, 2)
        with capture_native_bn(bn, connected_prefix=True) as statistics:
            bn(a)
        before = cpu_rng_state()
        with self.assertRaisesRegex(ValueError, "model RNG stream"):
            with auxiliary_bn(bn, captured=statistics):
                bn(normal)
                torch.rand(4)  # Deliberately inject an auxiliary stochastic op.
        self.assertFalse(rng_equal(before, cpu_rng_state()))
        self.assertNotIn("forward", bn.__dict__)

    def test_whole_a_batch_statistics_and_connected_derivatives(self):
        bn = nn.BatchNorm2d(2).double().train()
        a = torch.randn(3, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        n = torch.randn(2, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        weights = torch.linspace(.2, 1.7, n.numel(), dtype=torch.float64).reshape_as(n)
        before = cpu_rng_state()
        with capture_native_bn(bn, connected_prefix=True) as statistics:
            bn(a)
        stat = statistics[""][0]
        variance, mean = torch.var_mean(a, dim=(0, 2, 3), unbiased=False)
        self.assertTrue(torch.equal(stat.mean, mean))
        self.assertTrue(torch.equal(stat.biased_variance, variance))
        self.assertEqual(stat.elements_per_channel, 12)
        running = buffer_snapshot(bn)
        with auxiliary_bn(bn, captured=statistics):
            result = bn(n)
        self.assertTrue(buffers_equal(bn, running))
        self.assertTrue(rng_equal(before, cpu_rng_state()))
        reference = ((n - mean[None, :, None, None]) / (variance[None, :, None, None] + bn.eps).sqrt()
                     * bn.weight[None, :, None, None] + bn.bias[None, :, None, None])
        self.assertTrue(torch.allclose(result, reference, atol=1e-12, rtol=1e-12))
        parameters = (a, n, bn.weight, bn.bias)
        actual = torch.autograd.grad((result * weights).sum(), parameters, retain_graph=True)
        expected = torch.autograd.grad((reference * weights).sum(), parameters)
        for left, right in zip(actual, expected, strict=True):
            self.assertTrue(torch.allclose(left, right, atol=1e-10, rtol=1e-10))
        self.assertGreater(float(actual[0].norm()), 0.)
        self.assertEqual(int(bn.num_batches_tracked), 1)

    def test_exception_restores_bn_method_flags_and_buffers(self):
        bn = nn.BatchNorm2d(2).train()
        with capture_native_bn(bn, connected_prefix=True) as statistics:
            bn(torch.randn(2, 2, 2, 2, requires_grad=True))
        saved = buffer_snapshot(bn)
        normal = torch.randn(2, 2, 2, 2)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with auxiliary_bn(bn, captured=statistics):
                bn(normal)
                raise RuntimeError("injected")
        self.assertNotIn("forward", bn.__dict__)
        self.assertTrue(bn.training and bn.track_running_stats)
        self.assertTrue(buffers_equal(bn, saved))

    def test_detached_statistics_refused_by_connected_route(self):
        bn = nn.BatchNorm2d(2).train()
        with capture_native_bn(bn, connected_prefix=False) as statistics:
            bn(torch.randn(2, 2, 2, 2, requires_grad=True))
        normal = torch.randn(2, 2, 2, 2)
        with self.assertRaisesRegex(ValueError, "graph"):
            with auxiliary_bn(bn, captured=statistics):
                bn(normal)


if __name__ == "__main__":
    unittest.main()
