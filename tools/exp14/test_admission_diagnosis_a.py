import unittest

import numpy as np

from tools.exp14.admission_diagnosis_a import EDGE_FLAGS, crop_margin, edge_metrics, removal_count


class AdmissionDiagnosisATest(unittest.TestCase):
    def test_single_rule_is_evaluated_within_a_review_version(self):
        held = {
            "1": [{"reasons": ["A", "B"]}, {"reasons": ["C"]}],
            "2": [{"reasons": ["A"]}, {"reasons": ["B", "C"]}],
            "3": [{"reasons": ["A", "C"]}],
        }
        self.assertEqual(removal_count(held, {"A"}), 1)
        self.assertEqual(removal_count(held, {"A", "C"}), 3)

    def test_dark_margin_preserves_target(self):
        original = np.zeros((100, 120, 3), np.uint8)
        original[20:80, 20:100] = 150
        gt = np.zeros((100, 120), dtype=bool)
        gt[40:45, 50:55] = True
        invalid = np.zeros_like(gt)
        invalid[:, :2] = True
        self.assertEqual(crop_margin(invalid), [2, 0, 120, 100])
        result = edge_metrics(original, invalid, gt)
        self.assertEqual(result["invalid_pixels"], 200)
        self.assertEqual(result["crop_leaves_invalid_pixels"], 0)
        self.assertTrue(result["a1_dark_margin_screen_pass"])

    def test_visible_edge_and_target_overlap_block_crop(self):
        original = np.full((100, 120, 3), 100, np.uint8)
        gt = np.zeros((100, 120), dtype=bool)
        gt[40:45, :3] = True
        invalid = np.zeros_like(gt)
        invalid[:, :2] = True
        result = edge_metrics(original, invalid, gt)
        self.assertEqual(result["visible_original_invalid_pixels"], 200)
        self.assertEqual(result["gt_overlap_pixels"], 10)
        self.assertFalse(result["a1_dark_margin_screen_pass"])

    def test_interior_mask_region_is_not_a_margin_problem(self):
        original = np.zeros((100, 100, 3), np.uint8)
        invalid = np.zeros((100, 100), dtype=bool)
        invalid[30:33, 33:36] = True
        gt = np.zeros_like(invalid)
        gt[45:54, 47:51] = True
        result = edge_metrics(original, invalid, gt)
        self.assertEqual(result["inner_components"], 1)
        self.assertFalse(result["a1_dark_margin_screen_pass"])
        held = {"1": [{"reasons": list(EDGE_FLAGS)}]}
        self.assertEqual(removal_count(held, EDGE_FLAGS), 1)
        self.assertEqual(removal_count(held, {"NONFULL_FRAME_VALID_MASK"}), 0)


if __name__ == "__main__":
    unittest.main()
