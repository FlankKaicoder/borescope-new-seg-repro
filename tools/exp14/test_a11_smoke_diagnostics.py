import unittest
from pathlib import Path

from tools.exp14.a11_smoke_diagnostics import case_indices, validate_native_loss


class DiagnosticTest(unittest.TestCase):
    def test_frozen_batch_compositions_are_distinct_and_reproducible(self):
        paths = [Path(f"sample_{i}.png") for i in range(7)]
        rows = {p: ("baseline" if i < 3 else "derived", {}) for i, p in enumerate(paths)}
        mixed = case_indices(paths, rows, "original_repair_mixed", 4)
        self.assertEqual(mixed, case_indices(paths, rows, "original_repair_mixed", 4))
        self.assertEqual(len(set(mixed)), 4)
        self.assertEqual(sum(i < 3 for i in mixed), 2)
        empty = case_indices(paths, rows, "all_background", 4)
        self.assertEqual(set(empty), {3, 4, 5, 6})
        with self.assertRaisesRegex(ValueError, "membership"):
            case_indices(paths[:-1], rows, "all_background", 4)

    def test_background_requires_classification_signal_without_foreground(self):
        def branch(zero):
            return {"elements": 10, "zero": zero, "nonfinite": 0}

        branches = {"model.23.cv3.0": branch(2), "model.23.cv2.0": branch(10), "model.23.cv4.0": branch(10)}
        loss = {"cls_loss": 0.2, "seg_loss": 0.0, "box_loss": 0.0}
        validate_native_loss("all_background", loss, {"8": 0}, branches)
        with self.assertRaisesRegex(ValueError, "foreground-dependent loss"):
            validate_native_loss("all_background", {**loss, "seg_loss": 0.02}, {"8": 0}, branches)
        with self.assertRaisesRegex(ValueError, "classification signal"):
            validate_native_loss("all_background", loss, {"8": 0}, {**branches, "model.23.cv3.0": branch(10)})
        validate_native_loss("original_repair_mixed", {**loss, "seg_loss": 0.02}, {"8": 3}, branches)


if __name__ == "__main__":
    unittest.main()
