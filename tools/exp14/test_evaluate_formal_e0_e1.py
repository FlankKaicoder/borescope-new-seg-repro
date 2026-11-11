import unittest

import numpy as np

from tools.exp14.audit_formal_e0_e1 import coordinates, expected_lr
from tools.exp14.evaluate_formal_e0_e1 import FIXED_CONF, NMS_IOU, STANDARD_CONF, fixed_counts


class FrozenEvaluationTests(unittest.TestCase):
    def test_mask_and_box_counts_use_separate_matches(self):
        result = fixed_counts(np.array([0, 0, 1, 2]), np.array([0, 1, 1]),
                              np.array([True, False, True, False]), np.array([True, False, False, False]))
        self.assertEqual(result["overall"]["mask_fp"], 2)
        self.assertEqual(result["overall"]["box_fp"], 3)
        self.assertEqual(result["overall"]["mask_fn"], 1)
        self.assertEqual(result["per_class"][2]["mask_fp"], 1)
        self.assertEqual(result["overall"]["mask_precision"], 0.5)
        self.assertEqual((STANDARD_CONF, FIXED_CONF, NMS_IOU), (0.001, 0.25, 0.70))

    def test_matching_rejects_duplicate_true_positives(self):
        with self.assertRaisesRegex(ValueError, "More matched"):
            fixed_counts(np.array([0, 0]), np.array([0]), np.array([True, True]), np.array([False, False]))

    def test_first_warmup_step_and_growth_position(self):
        protocol = {"lr0": 0.01, "lrf": 0.01, "warmup_epochs": 3, "warmup_bias_lr": 0.1}
        self.assertEqual(expected_lr(0, protocol), [0.0, 0.0, 0.1])
        self.assertEqual(coordinates(1999), (91, 20))
        self.assertTrue(all(abs(x - 0.009703) < 1e-10 for x in expected_lr(66, protocol)))


if __name__ == "__main__":
    unittest.main()
