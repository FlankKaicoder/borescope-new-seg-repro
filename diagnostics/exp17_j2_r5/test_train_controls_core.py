"""CPU-only postprocessing fixtures, with no model construction or image reads."""
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from train_controls_core import LOGGER, NmsTimeoutError, native_match_pairs, overlap_instances, postprocess


def fixture(batch=1, wrong_class=False, no_detections=False):
    decoded = torch.zeros(batch, 12, 8, dtype=torch.float32)
    proto = torch.full((batch, 1, 160, 160), -1., dtype=torch.float32)
    proto[:, :, 40:120, 40:120] = 1.
    masks = torch.zeros(batch, 160, 160, dtype=torch.float32)
    masks[:, 40:120, 40:120] = 1.
    for index in range(batch):
        decoded[index, :4, 0] = torch.tensor([320., 320., 320., 320.])
        decoded[index, 4 + 2, 0] = 0. if no_detections else .8
        decoded[index, -1, 0] = 1.
        if wrong_class:
            decoded[index, :4, 1] = decoded[index, :4, 0]
            decoded[index, 4 + 5, 1] = .9
            decoded[index, -1, 1] = 1.
    data = {"img": torch.zeros(batch, 3, 640, 640), "cls": torch.full((batch, 1), 2.),
            "bboxes": torch.tensor([[.5, .5, .5, .5]]).repeat(batch, 1),
            "batch_idx": torch.arange(batch), "masks": masks}
    return decoded, proto, data


class PostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_overlap_ids_preserve_target_order(self):
        overlap = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.float32)
        actual = overlap_instances(overlap, 2)
        self.assertEqual(tuple(actual.shape), (2, 2, 3))
        self.assertEqual(actual[:, 0, 1].tolist(), [1., 0.])
        self.assertEqual(actual[:, 1, 0].tolist(), [0., 1.])
        self.assertEqual(actual.sum((1, 2)).tolist(), [2., 2.])
        with self.assertRaises(ValueError):
            overlap_instances(overlap, 1)

    def test_box_and_mask_matching_are_same_class_and_native(self):
        pairs = native_match_pairs(np.array([1, 0]), np.array([0, 1]), np.array([[1., .8], [.9, .7]]))
        self.assertEqual(set(map(tuple, pairs.tolist())), {(0, 1), (1, 0)})
        decoded, proto, data = fixture(wrong_class=True)
        before = decoded.clone()
        with tempfile.TemporaryDirectory() as temporary:
            result = postprocess(decoded, proto, data, temporary, "fixture")
            for key in (".001", ".25"):
                row = result[key]
                self.assertEqual((row["detections"], row["box_tp"], row["box_fp"], row["box_fn"]), (2, 1, 1, 0))
                self.assertEqual((row["mask_tp"], row["mask_fp"], row["mask_fn"]), (1, 1, 0))
                self.assertAlmostEqual(row["gt_mean_best_box_iou_ignoring_class"], 1., places=6)
                self.assertEqual(row["invalid_coordinates"]["nonfinite"], 0)
                with np.load(Path(temporary) / row["nms_predictions"]["file"]) as saved:
                    self.assertEqual(saved["image_000_predictions"].shape, (2, 7))
                    self.assertEqual(saved["image_000_box_matches"].shape, (1, 2))
                    self.assertEqual(saved["image_000_mask_matches"].shape, (1, 2))
            self.assertTrue(torch.equal(decoded, before))
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_detections_keep_null_precision(self):
        decoded, proto, data = fixture(no_detections=True)
        with tempfile.TemporaryDirectory() as temporary:
            result = postprocess(decoded, proto, data, temporary, "empty")
        for row in result.values():
            self.assertEqual((row["detections"], row["mask_tp"], row["mask_fn"]), (0, 0, 1))
            self.assertIsNone(row["mask_precision"])
            self.assertEqual(row["mask_recall"], 0.)
            self.assertIsNone(row["prediction_mean_best_box_iou_ignoring_class"])

    def test_nms_is_imagewise_without_rule_changes(self):
        from train_controls_core import nms
        decoded, proto, data = fixture(batch=2)
        original = nms.non_max_suppression
        calls = []

        def observe(prediction, **kwargs):
            calls.append((prediction.shape[0], kwargs))
            return original(prediction, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch.object(nms, "non_max_suppression", side_effect=observe):
            result = postprocess(decoded, proto, data, temporary, "batch")
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(size == 1 for size, _ in calls))
        self.assertTrue(all(kwargs["iou_thres"] == .7 and kwargs["max_det"] == 300
                            and kwargs["multi_label"] is True and kwargs["end2end"] is False for _, kwargs in calls))
        self.assertEqual(result[".25"]["mask_tp"], 2)

    def test_image_without_gt_keeps_false_positive(self):
        decoded, proto, data = fixture()
        data.update(cls=torch.empty(0, 1), bboxes=torch.empty(0, 4), batch_idx=torch.empty(0),
                    masks=torch.zeros(1, 160, 160, dtype=torch.uint8))
        with tempfile.TemporaryDirectory() as temporary:
            result = postprocess(decoded, proto, data, temporary, "no_gt")
        for row in result.values():
            self.assertEqual((row["gt"], row["detections"], row["box_tp"], row["box_fp"], row["mask_tp"], row["mask_fp"]),
                             (0, 1, 0, 1, 0, 1))
            self.assertIsNone(row["mask_recall"])

    def test_native_timeout_is_raised_and_handler_removed(self):
        from train_controls_core import nms
        decoded, proto, data = fixture()
        handlers = list(LOGGER.handlers)

        def timed_out(*args, **kwargs):
            LOGGER.warning("NMS time limit 2.050s exceeded")
            return [torch.zeros(0, 7)]

        with tempfile.TemporaryDirectory() as temporary, patch.object(nms, "non_max_suppression", side_effect=timed_out):
            with self.assertRaises(NmsTimeoutError):
                postprocess(decoded, proto, data, temporary, "timeout")
            self.assertEqual(list(Path(temporary).iterdir()), [])
        self.assertEqual(LOGGER.handlers, handlers)


if __name__ == "__main__":
    unittest.main()
