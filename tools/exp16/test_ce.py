"""CPU checks of CE eligibility, class reduction, geometry and RNG isolation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools.exp15.regions import Regions, add_projection, project_regions
from tools.exp16 import ce, train


def regions(classes, groups, cells=None):
    n = len(classes)
    return Regions(np.arange(n), np.zeros(n, dtype=np.int64), np.array(classes), np.array(groups),
                   np.ones(n, dtype=np.int64) if cells is None else np.array(cells), np.ones(n), 1)


class CETests(unittest.TestCase):
    def test_ce_uses_j1_eligibility_and_equal_class_means(self):
        r = regions([0, 0, 0, 1, 1, 2], ["a", "b", "c", "d", "e", "f"])
        value = torch.linspace(-1, 1, 6 * 128).reshape(6, 128).requires_grad_()
        head = nn.Linear(128, 7)
        actual = ce.region_cross_entropy(value, r, head)
        per = nn.functional.cross_entropy(head(value[:5]), torch.tensor([0, 0, 0, 1, 1]), reduction="none")
        expected = (per[:3].mean() + per[3:].mean()) / 2
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(int(value.grad[5].count_nonzero()), 0)
        self.assertGreater(int(value.grad[:5].count_nonzero()), 0)

    def test_same_group_missing_and_no_positive_policy_is_frozen(self):
        r = regions([0, 0, 1, 1, 1, 2], ["a", "a", "b", "c", "c", "d"], [1, 1, 1, 1, 1, 0])
        self.assertEqual(r.pairs()[2].tolist(), [False, False, True, True, True])
        value = torch.ones(5, 128, requires_grad=True)
        loss = ce.region_cross_entropy(value, r, nn.Linear(128, 7))
        loss.backward()
        self.assertEqual(int(value.grad[:2].count_nonzero()), 0)
        self.assertGreater(int(value.grad[2:].count_nonzero()), 0)
        with self.assertRaises(ValueError):
            ce.region_cross_entropy(torch.ones(2, 128), regions([0, 0], ["a", "a"]), nn.Linear(128, 7))

    def test_encoder_identity_and_native_rng(self):
        before = torch.get_rng_state().clone()
        supcon, cross_entropy = nn.Module(), nn.Module()
        add_projection(supcon)
        ce.add_head(cross_entropy)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(dict(cross_entropy.named_parameters())), 6)
        for left, right in zip(supcon.region_projection.parameters(), cross_entropy.region_projection[:3].parameters(), strict=True):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout))
                             for module in cross_entropy.region_projection.modules()))

    def test_same_polygon_pool_receives_ce_gradient_only_at_polygon(self):
        r = regions([0, 0, 1, 1], ["a", "b", "c", "d"])
        model = nn.Module()
        ce.add_head(model)
        feature = torch.rand(4, 64, 2, 2, requires_grad=True)
        mask = torch.zeros(4, 4, 4, dtype=torch.int64)
        mask[:, 0, 0] = 1
        embedding = project_regions(feature, mask, r, model.region_projection[:3])
        ce.region_cross_entropy(embedding, r, model.region_projection[3]).backward()
        self.assertGreater(int(feature.grad[:, :, 0, 0].count_nonzero()), 0)
        self.assertEqual(int(feature.grad[:, :, 1, :].count_nonzero()), 0)
        self.assertEqual(int(feature.grad[:, :, :, 1].count_nonzero()), 0)

    def test_wrong_seed_and_checkpoint_rejected_before_model_setup(self):
        with self.assertRaises(ValueError):
            train.make_trainer(None, None, None, None, None, 43)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "bad.pt"
            torch.save({"schema": "exp15_ja_train_smoke_v1"}, file)
            with self.assertRaises(ValueError):
                train.strict_checkpoint(file, 1)


if __name__ == "__main__":
    unittest.main()
