"""Test Exp15 J-B overlap binding and its separated optimization budgets."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.exp15 import jb
from tools.exp15.jb_binding import check_formatted, layout_counts, reconstruct


class JBTests(unittest.TestCase):
    def test_overlapping_disparate_polygons_require_joint_permutation(self):
        points = np.array([[[8, 8], [51, 8], [51, 51], [8, 51]],
                           [[13, 13], [47, 13], [47, 47], [13, 47]]])
        ids, order = reconstruct(points[::-1], 64, 64, 4)
        self.assertEqual(order.tolist(), [1, 0])
        self.assertEqual(int(ids[4, 4]), 2)
        self.assertEqual(int(ids[2, 2]), 1)
        categories = np.array([6, 1])
        boxes = np.array([[0.47, 0.47, 0.6, 0.6], [0.46, 0.46, 0.5, 0.5]], dtype=np.float32)
        native = {"masks": torch.from_numpy(ids[None]), "cls": torch.from_numpy(categories[:, None]),
                  "bboxes": torch.from_numpy(boxes)}
        self.assertEqual(layout_counts(boxes, categories)["high_box_iou_different_class_pairs"], 1)
        check_formatted(ids, categories, boxes, native)
        native["cls"] = native["cls"].flip(0)
        with self.assertRaises(ValueError):
            check_formatted(ids, categories, boxes, native)

    def test_budget_persists_and_cannot_overrun(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "j1.attempts").write_text("".join(f"{i}\n" for i in range(1, jb.SMOKE_CAP)), encoding="ascii")
            self.assertEqual(jb.reserve_attempt(path, jb.SMOKE_CAP, "j1"), jb.SMOKE_CAP)
            with self.assertRaises(ValueError):
                jb.reserve_attempt(path, jb.SMOKE_CAP, "j1")

    def test_formal_gate_rejects_absent_and_wrong_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            spec = jb.formal_protocol({"test": True}, "a", "b")
            self.assertEqual(spec["effective_updates_per_arm"], 2100)
            self.assertEqual(spec["lambda_s"]["j1"], 8)
            jb.ja.record_json(path / "protocol.json", spec)
            with self.assertRaises(ValueError):
                jb.verify_authorization(path / "protocol.json", path / "missing.json", spec["identity"])
            jb.ja.record_json(path / "forged.json", {"decision": "PASS", "protocol_sha256": "incorrect"})
            with self.assertRaises(ValueError):
                jb.verify_authorization(path / "protocol.json", path / "forged.json", spec["identity"])


if __name__ == "__main__":
    unittest.main()
