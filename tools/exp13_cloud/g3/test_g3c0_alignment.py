import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


SCRIPT = Path(__file__).with_name("align_g3b2_batch.py")
SPEC = importlib.util.spec_from_file_location("align_g3b2_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class G3C0GeometryTests(unittest.TestCase):
    def test_resize_candidate_requires_distributed_feature_support(self):
        source_size = (640, 352)
        output_size = (1280, 704)
        matrix = MODULE.resize_candidate(source_size, output_size)
        points = []
        for y in (70, 130, 210, 280):
            for x in (80, 200, 350, 520):
                points.append({"output": [x * 2, y * 2], "source": [x, y]})
        result = MODULE.validate_resize_candidate(points, matrix, source_size, output_size)
        self.assertEqual(result["status"], "GEOMETRY_CANDIDATE")
        self.assertEqual(result["inlier_count"], len(points))

    def test_resize_candidate_rejects_no_or_local_correspondence(self):
        source_size = (640, 352)
        output_size = (1280, 704)
        matrix = MODULE.resize_candidate(source_size, output_size)
        points = [{"output": [100 + i, 100 + i], "source": [50 + i / 2, 50 + i / 2]}
                  for i in range(20)]
        result = MODULE.validate_resize_candidate(points, matrix, source_size, output_size)
        self.assertEqual(result["status"], "ALIGNMENT_UNVERIFIED")

    def test_composites_never_change_outside_valid_target(self):
        original = np.full((40, 50, 3), 25, np.uint8)
        aligned = np.full_like(original, 220)
        target = np.zeros((40, 50), np.uint8)
        cv2.rectangle(target, (12, 10), (35, 29), 255, -1)
        valid = np.full((40, 50), 255, np.uint8)
        valid[:, :20] = 0
        target_valid, hard, feather = MODULE.composites(original, aligned, valid, target)
        allowed = (target_valid > 0)
        self.assertTrue(np.array_equal(hard[~allowed], original[~allowed]))
        self.assertTrue(np.array_equal(feather[~allowed], original[~allowed]))
        self.assertTrue(np.all(hard[allowed] == 220))
        self.assertEqual(cv2.countNonZero(target_valid), int(np.count_nonzero((target > 0) & (valid > 0))))

    def test_valid_support_marks_out_of_frame_pixels_invalid(self):
        matrix = np.float64([[1, 0, 10], [0, 1, 5]])
        valid = MODULE.valid_support(matrix, (20, 20), (30, 30))
        self.assertEqual(valid.shape, (30, 30))
        self.assertEqual(int(valid[0, 0]), 0)
        self.assertEqual(int(valid[15, 15]), 255)

    def test_rgb_image_round_trips_without_channel_swap(self):
        image = np.zeros((4, 5, 3), np.uint8)
        image[:, :, 0] = 210
        image[:, :, 1] = 80
        image[:, :, 2] = 15
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rgb.png"
            self.assertTrue(MODULE.write_rgb(path, image))
            self.assertTrue(np.array_equal(MODULE.read_rgb(path), image))


if __name__ == "__main__":
    unittest.main()
