"""Exact sparse-index adapter equivalence; no alternative Pair objective."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from tools.exp15.regions import Regions
from tools.exp17.pair_loss import source_ranking
from tools.exp17.engineering.losses import joint_loss, sparse_source_ranking


def regions(classes, groups):
    n = len(classes)
    return Regions(np.arange(n), np.zeros(n, dtype=np.int64), np.array(classes), np.array(groups),
                   np.ones(n, dtype=np.int64), np.ones(n), 1)


class LossTests(unittest.TestCase):
    def test_sparse_loss_and_gradients_equal_frozen_dense_reference(self):
        torch.manual_seed(19)
        a0 = torch.randn(6, 8, requires_grad=True); n0 = torch.randn(2, 8, requires_grad=True)
        a, n = F.normalize(a0, dim=1), F.normalize(n0, dim=1)
        ar = regions((0, 0, 0, 1, 1, 1), ("a", "b", "b", "c", "d", "e"))
        nr = regions((0, 1), ("a", "c"))
        bindings = [SimpleNamespace(a_row=0, n_row=0, group="a", class_id=0, identity="a"),
                    SimpleNamespace(a_row=3, n_row=1, group="c", class_id=1, identity="c")]
        loss, report = sparse_source_ranking(a, n, ar, nr, bindings)
        dense = torch.stack([n[0], n[0]*0, n[0]*0, n[1], n[1]*0, n[1]*0])
        old = [SimpleNamespace(**{**vars(item), "n_row": item.a_row}) for item in bindings]
        reference, _ = source_ranking(a, dense, ar, old, margin=.20)
        self.assertTrue(torch.equal(loss, reference)); self.assertEqual(report["paired_anchors"], 2)
        expected = torch.autograd.grad(reference, (a0, n0), retain_graph=True)
        actual = torch.autograd.grad(loss, (a0, n0))
        for g1, g2 in zip(expected, actual, strict=True): self.assertTrue(torch.equal(g1, g2))

    def test_missing_b_skips_without_fabrication_and_misbinding_is_rejected(self):
        ar = regions((0, 1), ("a", "b")); nr = regions((0,), ("a",))
        a = F.normalize(torch.randn(2, 8), dim=1); n = F.normalize(torch.randn(1, 8), dim=1)
        binding = SimpleNamespace(a_row=0, n_row=0, group="a", class_id=0, identity="a")
        loss, report = sparse_source_ranking(a, n, ar, nr, [binding])
        self.assertEqual(float(loss), 0); self.assertEqual(report["skipped"]["no_cross_source_same_class_B"], 1)
        bad = SimpleNamespace(**{**vars(binding), "group": "wrong"})
        with self.assertRaises(ValueError): sparse_source_ranking(a, n, ar, nr, [bad])
        with self.assertRaises(ValueError): sparse_source_ranking(a, n, ar, nr, [binding, binding])

    def test_only_explicit_lambda_and_frozen_supcon_weight(self):
        a, b, c = (torch.tensor(v, requires_grad=True) for v in (2., 3., 4.))
        self.assertEqual(float(joint_loss(a, b, c, .25).detach()), 27.)
        for invalid in (None, True, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): joint_loss(a, b, c, invalid)


if __name__ == "__main__": unittest.main()
