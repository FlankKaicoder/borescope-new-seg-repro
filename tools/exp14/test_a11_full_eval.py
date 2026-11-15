import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.exp14.a11_full_eval import (
    EVALUATION, GATE_SCHEMA, SIZE_THRESHOLDS, SizeStats, accepted_checkpoint,
    native_match_pairs, size_bin, val_source, validate_gate,
)
from tools.exp14.a11_full_protocol import protocol_spec
from tools.exp14.audit_admission import sha256_file


class FormalValEvaluationTest(unittest.TestCase):
    def test_preregistration_and_separate_val_gate_bind_exact_artifacts(self):
        self.assertEqual(protocol_spec()["evaluation"], EVALUATION)
        checkpoints = {"e0": "a", "e1": "b"}
        gate = {"schema": GATE_SCHEMA, "authorized": True, "protocol_sha256": "p",
                "identity_sha256": "i", "checkpoint_sha256": checkpoints}
        validate_gate(gate, "p", "i", checkpoints)
        for edited in ({**gate, "authorized": 1}, {**gate, "split": "test"}, {**gate, "checkpoint_sha256": {"e0": "a"}}):
            with self.assertRaisesRegex(ValueError, "VAL-only Gate"):
                validate_gate(edited, "p", "i", checkpoints)

    def test_native_greedy_dedup_and_class_filter(self):
        pred_cls = true_cls = np.array([0, 0, 1])
        iou = np.array([[0.90, 0.85, 0.99], [0.80, 0.40, 0.98], [0.95, 0.80, 0.50]])
        actual = native_match_pairs(pred_cls, true_cls, iou)
        np.testing.assert_array_equal(actual, [[0, 0], [2, 2]])
        self.assertEqual(native_match_pairs(pred_cls[:0], true_cls, np.zeros((3, 0))).shape, (0, 2))
        with self.assertRaisesRegex(ValueError, "frozen mapping"):
            native_match_pairs(np.array([0.5]), np.array([1]), np.ones((1, 1)))

    def test_strata_count_unmatched_targets_at_train_defined_boundaries(self):
        bins = SizeStats()
        gt = np.array([0, 0, 1, 1])
        areas = np.array([*SIZE_THRESHOLDS, 0.10])
        bins.add(gt, areas, np.array([[0, 2], [2, 0]]))
        report = bins.report()
        self.assertEqual([size_bin(a) for a in areas], ["tiny", "small", "medium", "large"])
        self.assertEqual([r["mask_recall"] for r in report["overall"].values()], [1.0, 0.0, 1.0, 0.0])
        self.assertIsNone(report["per_class"]["Crack"]["tiny"]["mask_recall"])
        self.assertEqual(sum(r["gt"] for r in report["overall"].values()), 4)
        self.assertEqual(sum(r["mask_tp"] for r in report["overall"].values()), 2)

    def test_last_receipt_rejects_partial_budget_and_changed_checkpoint(self):
        doc = protocol_spec()
        arm = doc["arms"]["e0"]
        updates = arm["effective_updates"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"synthetic final checkpoint identity")
            summary = root / "summary.json"
            receipt = {**dict.fromkeys(("optimizer_updates", "native_steps", "optimizer_attempts"), updates),
                       **dict.fromkeys(("amp_skips", "val_model_inference_runs", "test_model_inference_runs"), 0),
                       "status": "PASS_FULL_TRAIN_ARM", "arm": "e0", "protocol_sha256": "p", "identity_sha256": "i",
                       "training_performed": True, "backward_called": True,
                       "data_before": doc["identities"], "data_after": doc["identities"],
                       "checkpoint": {"sha256": sha256_file(checkpoint), "completed_epochs": 100,
                                      "optimizer_updates": updates, "strict_terminal_reload": True},
                       "counts": {"complete_epochs": 100,
                                  **dict.fromkeys(("batches_read", "forwards", "backwards", "finite_forwards", "completed_backwards"), updates),
                                  **{k: arm[k] for k in ("original_exposures", "repair_exposures", "raw_instance_exposures")}}}
            summary.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(accepted_checkpoint("e0", root, doc, "p", "i")[0], checkpoint)
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "last.pt identity"):
                accepted_checkpoint("e0", root, doc, "p", "i")
            receipt["counts"]["complete_epochs"] = 99
            summary.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "100-epoch"):
                accepted_checkpoint("e0", root, doc, "p", "i")

    def test_val_member_identity_rejects_test_and_wrong_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            image = root / "images/val/one.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"synthetic VAL member")
            row = {"split": "val", "stem": "one", "image_suffix": ".png", "image_path": image.name,
                   "sha256": sha256_file(image)}
            self.assertEqual(val_source(root, row, "image", None)[:2], (image, image))
            with self.assertRaisesRegex(ValueError, "Only VAL"):
                val_source(root, {**row, "split": "test"}, "image", None)
            with self.assertRaisesRegex(ValueError, "declared identity"):
                val_source(root, {**row, "sha256": "a" * 64}, "image", None)


if __name__ == "__main__":
    unittest.main()
