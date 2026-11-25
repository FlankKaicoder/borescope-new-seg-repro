import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.exp18.bn import cpu_rng_state, rng_equal
from tools.exp18.model import (AForward, a0_loss, a1_loss, add_state_head,
                               forward_joint, native_state_dict, pool_raw_regions)


class ToyGraph(nn.Module):
    exp18_synthetic = True

    def __init__(self, *, state_head=True):
        super().__init__()
        self.shared = nn.Sequential(nn.Conv2d(3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.region_projection = nn.Linear(64, 64)
        self.native_head = nn.Linear(64, 1)
        if state_head:
            add_state_head(self, seed=180042)


def regions(count=2):
    return SimpleNamespace(cells=np.ones(count, dtype=np.int64), slots=np.arange(count),
                           ordinals=np.zeros(count, dtype=np.int64), max_instances=1)


def callbacks():
    def a_forward(model, batch, region):
        feature = model.shared(batch["img"])
        raw = pool_raw_regions(feature, batch["masks"], region)
        yolo = model.native_head(raw).square().mean()
        # Synthetic loss supplies a real gradient for all projection parameters.
        supcon = model.region_projection(raw).square().mean()
        return AForward(yolo, supcon, feature)
    def n_forward(model, image):
        return model.shared(image)
    return a_forward, n_forward


class ModelTests(unittest.TestCase):
    def test_state_head_rng_bias_and_shared_initialization(self):
        base = nn.Linear(3, 2)
        first, second = copy.deepcopy(base), copy.deepcopy(base)
        before = cpu_rng_state()
        add_state_head(first, seed=180042)
        self.assertTrue(rng_equal(before, cpu_rng_state()))
        add_state_head(second, seed=180042)
        self.assertTrue(torch.equal(first.state_head.weight, second.state_head.weight))
        self.assertIsNone(first.state_head.bias)
        self.assertGreater(int(first.state_head.weight.count_nonzero()), 0)
        self.assertTrue(torch.equal(first.weight, base.weight))

    def test_a0_invariant_and_a1_sensitive_with_fixed_marginals(self):
        head = nn.Linear(64, 1, bias=False).double()
        with torch.no_grad():
            head.weight.zero_(); head.weight[0, 0] = 8.
        def features(scores):
            value = torch.zeros(2, 64, dtype=torch.float64)
            value[:, 0] = torch.tensor(scores, dtype=torch.float64) / 8.
            value[:, 1] = (1 - value[:, 0].square()).sqrt()
            return value
        a, n = features([2., 4.]), features([1., 3.])
        weights = [.5, .5]
        self.assertAlmostEqual(float(a0_loss(head, a, n, weights)), float(a0_loss(head, a, n.flip(0), weights)), places=14)
        self.assertLess(float(a1_loss(head, a, n, weights)), float(a1_loss(head, a, n.flip(0), weights)))
        self.assertAlmostEqual(float((head(a) - head(n)).mean()), float((head(a) - head(n.flip(0))).mean()), places=14)

    def test_a1_zero_head_first_gradient_is_invariant(self):
        head = nn.Linear(64, 1, bias=False).double()
        with torch.no_grad():
            head.weight.zero_()
        a = F.normalize(torch.randn(3, 64, dtype=torch.float64), dim=1)
        n = F.normalize(torch.randn(3, 64, dtype=torch.float64), dim=1)
        first = torch.autograd.grad(a1_loss(head, a, n, [1/3] * 3), head.weight)[0]
        second = torch.autograd.grad(a1_loss(head, a, n[[1, 2, 0]], [1/3] * 3), head.weight)[0]
        self.assertTrue(torch.allclose(first, second, atol=1e-14, rtol=1e-14))

    def test_own_roi_pooling_precedes_permutation(self):
        # ROI0 uses top-left, ROI1 bottom-right. A cross-image mask swap would fail.
        feature = torch.zeros(2, 64, 2, 2, dtype=torch.float64)
        feature[0, 0, 0, 0], feature[0, 2, 1, 1] = 1., 9.
        feature[1, 1, 1, 1], feature[1, 3, 0, 0] = 1., 9.
        masks = torch.zeros(2, 4, 4, dtype=torch.long)
        masks[0, :2, :2], masks[1, 2:, 2:] = 1, 1
        pooled = pool_raw_regions(feature, masks, regions())
        self.assertEqual(pooled.argmax(1).tolist(), [0, 1])
        self.assertEqual(pooled[[1, 0]].argmax(1).tolist(), [1, 0])

    def test_joint_n_gradient_buffers_empty_and_native_view(self):
        model = ToyGraph().train()
        batch = {"img": torch.randn(2, 3, 2, 2), "masks": torch.ones(2, 4, 4, dtype=torch.long)}
        normal = torch.randn(2, 3, 2, 2, requires_grad=True)
        plan = {"a_rows": (0, 1), "n_rows": (0, 1), "permutation": (1, 0),
                "weights": (.5, .5), "sha256": "a" * 64}
        a_forward, n_forward = callbacks()
        bn = model.shared[1]
        before = int(bn.num_batches_tracked)
        result = forward_joint(model, batch, regions(), normal, batch["masks"], regions(), plan,
                               arm="a1_true", lambda_state=1., a_forward=a_forward, n_forward=n_forward)
        self.assertEqual(int(bn.num_batches_tracked) - before, 1)
        result.total.backward()
        self.assertTrue(result.head_active)
        self.assertIsNotNone(normal.grad)
        self.assertGreater(float(normal.grad.norm()), 0.)
        self.assertEqual(tuple(result.n_raw.shape), (2, 64))
        self.assertTrue(all(not k.startswith(("region_projection.", "state_head.")) for k in native_state_dict(model)))
        model.zero_grad(set_to_none=True)
        empty = {"a_rows": (), "n_rows": (), "permutation": (), "weights": (), "sha256": "b" * 64}
        result = forward_joint(model, batch, regions(), None, None, None, empty,
                               arm="a1_shuffled", lambda_state=1., a_forward=a_forward, n_forward=n_forward)
        result.total.backward()
        self.assertFalse(result.head_active)
        self.assertIsNone(model.state_head.weight.grad)

    def test_j1_has_no_state_head_and_no_n_traversal(self):
        model = ToyGraph(state_head=False).train()
        batch = {"img": torch.randn(2, 3, 2, 2), "masks": torch.ones(2, 4, 4, dtype=torch.long)}
        a_forward, _ = callbacks()
        def forbidden(*args):
            raise AssertionError("J1 executed N")
        result = forward_joint(model, batch, regions(), None, None, None, None,
                               arm="j1", lambda_state=1., a_forward=a_forward, n_forward=forbidden)
        result.total.backward()
        self.assertFalse(result.head_active)
        self.assertFalse(hasattr(model, "state_head"))


if __name__ == "__main__":
    unittest.main()
