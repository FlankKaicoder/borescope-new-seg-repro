"""Synthetic CPU-only checks; no checkpoint or model construction."""
import unittest

import numpy as np

from tools.exp18.geometry import (
    coverage_weights, is_complete_valid_support, native_polygon_area_fraction,
    native_polygon_geometry, region_geometry, stride4_valid_mask,
)


class GeometryTests(unittest.TestCase):
    def test_native_fraction_uses_native_float32_and_no_source_size(self):
        points = [0.1, 0.2, 0.6, 0.2, 0.6, 0.7, 0.1, 0.7]
        coordinates = np.asarray(points, dtype=np.float32).astype(np.float64).reshape(-1, 2)
        expected = (coordinates[1, 0] - coordinates[0, 0]) * (coordinates[2, 1] - coordinates[1, 1])
        self.assertAlmostEqual(native_polygon_area_fraction(points), expected, places=15)
        self.assertEqual(native_polygon_area_fraction(points), native_polygon_area_fraction(coordinates[::-1]))

    def test_invalid_native_geometry_is_not_repaired(self):
        cases = ([0, 0, 1], [[0, 0], [0, 0], [0, 0]],
                 [[0, 0], [1.1, 0], [0, 1]], [[0, 0], [float("nan"), 0], [0, 1]])
        for points in cases:
            with self.subTest(points=points), self.assertRaises(ValueError):
                native_polygon_area_fraction(points)

    def test_fractional_coverage_keeps_tiny_roi_without_interpolation(self):
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[10, 11] = 1
        raw, normalized = coverage_weights(mask)
        self.assertEqual(raw.shape, (80, 80))
        self.assertEqual(np.count_nonzero(raw), 1)
        self.assertEqual(float(raw[5, 5]), .25)
        self.assertEqual(float(normalized[5, 5]), 1.)
        self.assertEqual(float(normalized.sum()), 1.)

    def test_cells_coverage_mass_and_effective_cells_are_distinct(self):
        overlap = np.zeros((4, 4), dtype=np.uint8)
        overlap[0:2, 0:2] = 1
        overlap[0, 2] = 1
        geometry = region_geometry(overlap, 1)
        self.assertEqual(geometry.cells, 2)
        self.assertEqual(geometry.native_mask_pixels, 5)
        self.assertEqual(geometry.coverage_mass, 1.25)
        np.testing.assert_array_equal(geometry.coverage_weights, [[1., .25], [0., 0.]])
        np.testing.assert_allclose(geometry.normalized_weights, [[.8, .2], [0, 0]], atol=1e-7)
        self.assertAlmostEqual(geometry.effective_cells, 1 / (.8**2 + .2**2), places=6)
        self.assertNotEqual(geometry.effective_cells, geometry.cells)
        self.assertIsNone(geometry.support_complete)
        self.assertEqual(geometry.to_dict()["p3_shape"], [2, 2])
        self.assertFalse(geometry.coverage_weights.flags.writeable)

    def test_valid_qualification_is_whole_foreground_not_whole_p3_cell(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 1
        valid = np.full((16, 16), 255, dtype=np.uint8)
        # Invalid support in another stride-4 cell of the same P3 cell does
        # not silently change the original foreground qualification rule.
        valid[7, 7] = 0
        self.assertTrue(is_complete_valid_support(mask, valid))
        valid[0, 0] = 0
        self.assertFalse(is_complete_valid_support(mask, valid))
        geometry = region_geometry(mask, 1, valid)
        self.assertFalse(geometry.support_complete)
        self.assertEqual(geometry.cells, 1)
        self.assertEqual(geometry.coverage_mass, .25)  # No invalid-pixel crop.

    def test_one_invalid_pixel_invalidates_its_entire_stride4_cell(self):
        valid = np.ones((8, 8), dtype=np.uint8)
        valid[1, 1] = 0
        np.testing.assert_array_equal(stride4_valid_mask(valid), [[False, True], [True, True]])

    def test_missing_native_rasters_are_reported_when_target_counts_supplied(self):
        overlap = np.zeros((1, 4, 4), dtype=np.uint8)
        overlap[0, 0, 0] = 1
        result = native_polygon_geometry(overlap, instances_per_image=[2], valid_full=np.ones((1, 16, 16)))
        self.assertEqual(set(result), {(0, 1), (0, 2)})
        self.assertEqual(result[(0, 2)].cells, 0)
        self.assertEqual(result[(0, 2)].effective_cells, 0.)
        self.assertFalse(result[(0, 2)].support_complete)
        np.testing.assert_array_equal(result[(0, 2)].normalized_weights, np.zeros((2, 2)))

    def test_overlap_ids_remain_disjoint(self):
        overlap = np.asarray([[1, 1, 2, 2], [1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
        result = native_polygon_geometry(overlap, instances_per_image=[2])
        self.assertEqual(result[(0, 1)].native_mask_pixels, 3)
        self.assertEqual(result[(0, 2)].native_mask_pixels, 3)
        self.assertNotEqual(result[(0, 1)].mask_sha256, result[(0, 2)].mask_sha256)

    def test_native_integral_float_ids_are_supported_without_mutating_bytes(self):
        overlap = np.asarray([[1., 1., 2., 2.], [1., 0., 2., 0.],
                              [0., 0., 0., 0.], [0., 0., 0., 0.]], dtype=np.float32)
        before = overlap.tobytes()
        result = native_polygon_geometry(overlap, instances_per_image=[2])
        self.assertEqual(result[(0, 1)].native_mask_pixels, 3)
        self.assertEqual(region_geometry(overlap, 2).cells, 1)
        self.assertEqual(overlap.dtype, np.dtype("float32"))
        self.assertEqual(overlap.tobytes(), before)

    def test_fractional_nonfinite_or_negative_float_ids_are_rejected(self):
        for value in (.25, float("nan"), float("inf"), -1.):
            mask = np.zeros((4, 4), dtype=np.float32)
            mask[0, 0] = value
            before = mask.tobytes()
            for call in (lambda: region_geometry(mask, 1), lambda: native_polygon_geometry(mask)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    call()
            self.assertEqual(mask.tobytes(), before)

    def test_invalid_arrays_are_rejected(self):
        calls = (lambda: coverage_weights(np.ones((3, 4))),
                 lambda: coverage_weights(np.full((4, 4), .5)),
                 lambda: stride4_valid_mask(np.ones((5, 8))),
                 lambda: region_geometry(np.full((4, 4), .25, dtype=np.float32), 1),
                 lambda: native_polygon_geometry(np.full((4, 4), 3, dtype=np.uint8), [2]),
                 lambda: is_complete_valid_support(np.ones((4, 4)), np.ones((8, 8))))
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
