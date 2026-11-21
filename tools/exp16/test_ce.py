"""CPU checks of CE eligibility, class reduction, geometry and RNG isolation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools.exp15.regions import Regions, add_projection, project_regions
from tools.exp16 import ce, train, experiment


def regions(classes, groups, cells=None):
    n = len(classes)
    return Regions(np.arange(n), np.zeros(n, dtype=np.int64), np.array(classes), np.array(groups),
                   np.ones(n, dtype=np.int64) if cells is None else np.array(cells), np.ones(n), 1)


class CETests(unittest.TestCase):
    def test_ce_uses_all_valid_regions_and_equal_class_means(self):
        r = regions([0, 0, 0, 1, 1, 2], ["a", "b", "c", "d", "e", "f"])
        value = torch.linspace(-1, 1, 6 * 128).reshape(6, 128).requires_grad_()
        head = nn.Linear(128, 7)
        actual = ce.region_cross_entropy(value, r, head)
        per = nn.functional.cross_entropy(head(value), torch.tensor([0, 0, 0, 1, 1, 2]), reduction="none")
        expected = (per[:3].mean() + per[3:5].mean() + per[5]) / 3
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(int(value.grad[5].count_nonzero()), 0)
        self.assertGreater(int(value.grad[:5].count_nonzero()), 0)

    def test_same_group_and_singleton_labels_receive_ce_but_missing_rasters_do_not(self):
        r = regions([0, 0, 1, 1, 1, 2], ["a", "a", "b", "c", "c", "d"], [1, 1, 1, 1, 1, 0])
        self.assertEqual(r.pairs()[2].tolist(), [False, False, True, True, True])
        value = torch.ones(5, 128, requires_grad=True)
        loss = ce.region_cross_entropy(value, r, nn.Linear(128, 7))
        loss.backward()
        self.assertGreater(int(value.grad[:2].count_nonzero()), 0)
        self.assertGreater(int(value.grad[2:].count_nonzero()), 0)
        self.assertEqual(ce.coverage(r)["valid_regions"], 5)
        self.assertEqual(ce.coverage(r)["ce_regions_by_class"], [2, 3, 0, 0, 0, 0, 0])
        self.assertTrue(torch.isfinite(ce.region_cross_entropy(
            torch.ones(2,128),regions([0,0],["a","a"]),nn.Linear(128,7))))
        with self.assertRaises(ValueError):
            ce.region_cross_entropy(torch.empty(0, 128), regions([0, 0], ["a", "a"], [0,0]), nn.Linear(128, 7))

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

    def test_diagnostic_scaling_unscales_and_actual_joint_recheck(self):
        model=nn.Module()
        model.model=nn.Sequential(nn.Linear(2,2))
        model.region_projection=nn.Sequential(nn.Linear(2,7))
        x=torch.tensor([[.2,.5],[.4,-.1]])
        labels=torch.tensor([0,1])
        features=model.model(x)
        native=features.square().sum()
        auxiliary=nn.functional.cross_entropy(model.region_projection(features),labels)
        expected=torch.autograd.grad(auxiliary,list(model.model.parameters()),retain_graph=True)
        expected_norm=sum(float(g.square().sum()) for g in expected)**.5
        probe=experiment.gradients(model,native,auxiliary)
        self.assertAlmostEqual(probe["backbone"]["auxiliary_norm"],expected_norm,places=6)
        analytical={key:experiment.weighted(probe[key],3.0) for key in ("backbone","global")}
        features=model.model(x)
        native=features.square().sum()
        auxiliary=nn.functional.cross_entropy(model.region_projection(features),labels)
        actual=experiment.actual_joint_probe(model,native,auxiliary,3.0,analytical)
        for scope in ("backbone","global"):
            self.assertLess(actual[scope]["analytic_joint_norm_error_over_native"],1e-5)


if __name__ == "__main__":
    unittest.main()
