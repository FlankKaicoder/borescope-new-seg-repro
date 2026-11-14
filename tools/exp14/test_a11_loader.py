import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools.exp14.a11_loader import batch_counts, read_expected_labels, validate_dataset


class DataLoaderEvidenceTest(unittest.TestCase):
    def test_native_records_must_keep_exact_polygon_and_class(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image, label = root / "sample.png", root / "sample.txt"
            image.write_bytes(b"test first bytes")
            label.write_text("2 0.25 0.25 0.5 0.25 0.5 0.5\n", encoding="utf-8")
            want = read_expected_labels(root, [{"image_path": image.name, "label_path": label.name, "instance_count": 1}])

            class Dataset:
                def __len__(self):
                    return 1

            data = Dataset()
            data.labels = [{"im_file": image, "cls": np.array([[2]], dtype=np.float32),
                            "segments": [np.array([[0.25, 0.25], [0.5, 0.25], [0.5, 0.5]], dtype=np.float32)]}]
            self.assertEqual(validate_dataset(data, want), {image: (2,)})
            data.labels[0]["segments"][0][0, 0] = 0
            with self.assertRaisesRegex(ValueError, "Polygon"):
                validate_dataset(data, want)

    def test_collate_routes_targets_and_checks_empty_masks(self):
        original, bg = Path("o.png"), Path("r.png")
        masks, sem_masks = np.zeros((2, 2, 2)), np.zeros((2, 2, 2))
        masks[0, 0, 0] = 1
        classes = np.asarray([[2]])
        indexes = np.asarray([0])
        boxes = np.ones((1, 4))
        self.assertEqual(
            batch_counts([original, bg], {original: (2,), bg: ()}, classes, indexes, boxes, masks, sem_masks, 2, 2),
            ("original_repair_mixed", 1, 1),
        )
        masks[1, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "background"):
            batch_counts([original, bg], {original: (2,), bg: ()}, classes, indexes, boxes, masks, sem_masks, 2, 2)
        with self.assertRaisesRegex(ValueError, "another image"):
            batch_counts([original, bg], {original: (2,), bg: ()}, classes, np.asarray([1]), boxes,
                         masks, sem_masks, 2, 2)
        self.assertEqual(batch_counts(
            [bg], {bg: ()}, np.empty((0, 1)), np.empty(0), np.empty((0, 4)), np.zeros((1, 2, 2)),
            np.zeros((1, 2, 2)), 1, 2), ("all_background", 0, 1))

    def test_reject_rewrite_hazard_and_nonexplicit_empty_label(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image, label = root / "bad.jpg", root / "bad.txt"
            row = {"image_path": image.name, "label_path": label.name, "instance_count": 0}
            image.write_bytes(b"\xff\xd8invalid")
            label.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "rewrite"):
                read_expected_labels(root, [row])
            image.write_bytes(b"\xff\xd8\xff\xd9")
            label.write_bytes(b"\n")
            with self.assertRaisesRegex(ValueError, "explicit empty"):
                read_expected_labels(root, [row])


if __name__ == "__main__":
    unittest.main()
