import json
import tempfile
import unittest
from pathlib import Path

from tools.exp13_cloud.g3.g3_batch_executor import digest, write_exclusive, write_json_exclusive
from tools.exp14.a10_freeze import bind_exceptions, content_receipt
from tools.exp14.a10_selection import FINAL


class FreezeTest(unittest.TestCase):
    def test_content_receipt_is_portable_and_requires_exact_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ("images/base/one.png", "labels/base/one.txt")
            image, label = b"image bytes", b"label bytes"
            for name, payload in zip(paths, (image, label)):
                write_exclusive(root / "dataset" / name, payload)
            entry = {"image_path": paths[0], "image_sha256": digest(image),
                     "label_path": paths[1], "label_sha256": digest(label)}
            write_json_exclusive(root / "configs" / "train_entry_files.json", {"baseline": [entry], "derived": []})
            receipt = content_receipt(root)
            self.assertEqual(receipt["file_count"], 2)
            self.assertEqual(receipt["sha256"], digest(json.dumps(receipt["files"], sort_keys=True,
                             separators=(",", ":")).encode("utf-8")))
            write_exclusive(root / "dataset" / "unexpected.cache", b"")
            with self.assertRaisesRegex(ValueError, "unexpected file"):
                content_receipt(root)

    def test_label_conflict_cannot_be_transferred_to_another_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = []
            for i in range(2):
                file = root / f"evidence_{i}"
                write_exclusive(file, b"evidence")
                evidence.append({"path": str(file), "sha256": digest(b"evidence")})
            image = str(root / "selected.png")
            selection = {"source_id": "s", "path": image, "sha256": "a", "original": {"sha256": "o"},
                         "annotation": {"sha256": "g"}, "asset_id": "id", "asset_type": "ALIGNED",
                         "review_id": "r", "variant_id": "v", "decision_origin": FINAL}
            exception = {"source_id": "s", "selected_asset_path": image, "asset_sha256": "a",
                         "original_sha256": "o", "annotation_sha256": "g", "asset_id": "id",
                         "asset_type": "ALIGNED", "review_id": "r", "variant_id": "v", "polygon_ids": ["p"],
                         "reason_code": "RESIDUAL_ANNOTATED_DEFECT_EXACT_ASSET", "evidence": evidence, "reason": "Residual"}
            inputs = {"input_csv_sha256": "input"}
            document = {"selected_input_sha256": "input", "entries": [exception]}
            a9 = {"sources": {"s": {"all_polygon_ids": ["p"]}}}
            self.assertEqual(bind_exceptions(document, inputs, [selection], a9, root, root), {"s": exception})
            exception["asset_sha256"] = "other"
            with self.assertRaisesRegex(ValueError, "exact human-selected pixels"):
                bind_exceptions(document, inputs, [selection], a9, root, root)
