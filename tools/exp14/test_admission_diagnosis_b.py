import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.exp14.admission_diagnosis_b import assess, eligible_pixel_support, region_metrics


class BRegionGateTests(unittest.TestCase):
    def test_boundary_clipping_keeps_context_and_target_denominators_separate(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[0:10, 0:10] = True
        valid = np.ones_like(mask)
        valid[10:15, :10] = False

        metrics = region_metrics(mask, valid)

        self.assertEqual(metrics["roi"], [0, 0, 22, 22])
        self.assertEqual(metrics["context_margin_pixels_ltrb"], [0, 0, 12, 12])
        self.assertEqual(metrics["unsupported_target_pixels"], 0)
        self.assertEqual(metrics["valid_roi_pixels"], 484 - 50)
        self.assertEqual(metrics["valid_context_pixels"], 384 - 50)
        self.assertTrue(metrics["frame_edge_touch"])
        self.assertFalse(eligible_pixel_support(metrics))

    def test_exact_pixel_gate_does_not_admit_rounded_up_percentage(self):
        metrics = {
            "unsupported_target_pixels": 0,
            "valid_roi_pixels": 1_899_999,
            "roi_pixels": 2_000_000,
            "valid_context_pixels": 100,
            "context_pixels": 100,
        }
        self.assertFalse(eligible_pixel_support(metrics))
        self.assertTrue(eligible_pixel_support({**metrics, "valid_roi_pixels": 1_900_000}))

    def test_dense_valid_mask_without_visual_measurement_is_hold(self):
        mask = np.ones((40, 40), dtype=bool)
        mask[:8, :] = False
        case = {"metrics": region_metrics(mask, np.ones_like(mask))}
        observation = ("YES", "YES", "YES", "CLEAR", "CLEAR", "fixture", [])

        decision = assess(case, observation, Path("."))

        self.assertTrue(decision["numeric_integrity_screen"])
        self.assertEqual(decision["g1_result"], "G1_HOLD_INDEPENDENT_BOUNDARY_EVIDENCE")
        self.assertFalse(decision["coarse_geometry_qualified_formally"])

    def test_axis_displacement_and_target_scale_leave_formal_gate_pending(self):
        mask = np.zeros((60, 80), dtype=bool)
        mask[20:40, 20:50] = True
        case = {"metrics": region_metrics(mask, np.ones_like(mask))}
        sample = {"kind": "independent", "station_axis": "x", "station_percent": 30,
                  "original_percent": 20, "aligned_percent": 24, "sheet": "visual.png", "feature": "boundary"}
        second = {**sample, "station_axis": "y", "station_percent": 50, "aligned_percent": 21}
        observation = ("YES", "YES", "YES", "CLEAR", "CLEAR", "fixture", [sample, second])

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "visual.png").write_bytes(b"synthetic evidence fixture")
            decision = assess(case, observation, folder)
            rejected = assess(case, (*observation[:4], "MATERIAL", *observation[5:]), folder)

        self.assertEqual(decision["g1_result"], "G1_PILOT_FEASIBLE_PENDING_INDEPENDENT_REVIEW")
        self.assertEqual(rejected["g1_result"], "G1_PILOT_FAIL")
        self.assertFalse(decision["training_approved"])
        self.assertEqual(decision["observed_max_displacement_px"], 44 * 0.04)
        self.assertAlmostEqual(decision["g1_displacement_limit_px"], 0.1 * math.hypot(44, 58), places=3)
        self.assertNotIn("station_evidence_sha256", sample)
