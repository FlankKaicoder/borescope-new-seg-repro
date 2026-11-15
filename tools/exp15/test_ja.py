"""Focused checks for group masks, native mask geometry and optimizer budget."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ultralytics.nn.tasks import SegmentationModel

from tools.exp15 import ja
from tools.exp15.regions import add_projection, inspect_regions, pair_masks, project_regions, supervised_contrast


class RegionTests(unittest.TestCase):
    def test_pair_policy(self):
        positive, negative, active = pair_masks(np.array([0, 0, 0, 1, 1]), np.array(["a", "a", "b", "a", "c"]))
        self.assertEqual(int(positive.sum()), 6)
        self.assertFalse(positive[0, 1])
        self.assertFalse(negative[0, 3])
        self.assertTrue(negative[0, 4])
        self.assertTrue(active.all())

    def test_pool_ignores_off_polygon_and_differentiates(self):
        batch = {
            "im_file": [__file__, ja.__file__, np.__file__], "img": torch.zeros(3, 3, 16, 16, dtype=torch.uint8),
            "batch_idx": torch.arange(3), "cls": torch.tensor([[0.], [0.], [1.]]),
            "bboxes": torch.tensor([[0.0625, 0.0625, 0.125, 0.125]] * 3),
            "masks": torch.zeros(3, 4, 4, dtype=torch.int64),
        }
        batch["masks"][:, 0, 0] = 1
        region = inspect_regions(batch, {str(Path(x).resolve()): x for x in batch["im_file"]})
        self.assertEqual(region.cells.tolist(), [1, 1, 1])
        generator = torch.Generator().manual_seed(11)
        feature = torch.rand((3, 64, 2, 2), generator=generator, requires_grad=True)
        embedding = project_regions(feature, batch["masks"], region, nn.Identity())
        torch.testing.assert_close(embedding, nn.functional.normalize(feature[:, :, 0, 0], dim=1))
        loss = supervised_contrast(embedding, region.classes, region)
        loss.backward()
        self.assertGreater(float(feature.grad[:, :, 0, 0].abs().sum()), 0)
        self.assertEqual(int(feature.grad[:, :, 1, :].count_nonzero()), 0)
        self.assertEqual(int(feature.grad[:, :, :, 1].count_nonzero()), 0)
        batch["masks"][0, 3, 3] = 1
        with self.assertRaises(ValueError):
            inspect_regions(batch, {str(Path(x).resolve()): x for x in batch["im_file"]})

    def test_projection_rng_and_inference(self):
        torch.set_num_threads(2)
        model = SegmentationModel("yolo11n-seg.yaml", nc=7, verbose=False).eval()
        x = torch.zeros(1, 3, 64, 64)
        rng = torch.get_rng_state().clone()
        add_projection(model)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        with torch.no_grad():
            attached, _ = model(x)
            native = ja.native_state(model)
            del model.region_projection
            stripped, _ = model(x)
        self.assertTrue(ja.exact_tree(dict(model.state_dict()), native))
        self.assertTrue(ja.exact_tree(attached, stripped))

    def test_attempt_cap_is_cumulative_and_per_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "j0.attempts").write_text("1\n" * (ja.MAX_ATTEMPTS - 1))
            ja.reserve_attempt(path, "j0")
            with self.assertRaises(ValueError):
                ja.reserve_attempt(path, "j0")
            ja.reserve_attempt(path, "j1")
            self.assertEqual(len((path / "j1.attempts").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()

