import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.exp13_cloud.g3 import build_candidate_review, verify_train_export


class G3ToolsTest(unittest.TestCase):
    def test_export_verify_and_candidate_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            image = source / "1.png"
            annotation = source / "1.json"
            Image.new("RGB", (32, 24), (120, 120, 120)).save(image)
            annotation.write_text(json.dumps({"imageWidth": 32, "imageHeight": 24, "shapes": [{"label": "Crack", "shape_type": "polygon", "points": [[4, 4], [12, 4], [8, 12]]}]}), encoding="utf-8")
            manifest = root / "manifest.csv"
            fields = ["image_path", "json_path", "stem", "split", "group_id", "sha256", "json_sha256", "labels_present", "instance_count", "image_width", "image_height"]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"image_path": "1.png", "json_path": "1.json", "stem": "1", "split": "train", "group_id": "singleton_1", "sha256": verify_train_export.sha256_file(image), "json_sha256": verify_train_export.sha256_file(annotation), "labels_present": "Crack", "instance_count": "1", "image_width": "32", "image_height": "24"})
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            export = root / "export"
            record = verify_train_export.export_train(argparse_namespace(mode="export", manifest=manifest, source_root=source, output=export, image_root=None, json_root=None, class_mapping=None, expected_manifest_sha256=manifest_sha, expected_count=1))
            self.assertEqual(record["train_images"], 1)
            verified = verify_train_export.verify(argparse_namespace(mode="verify", manifest=export / "split_manifest.csv", source_root=export, output=root / "verify.json", image_root=None, json_root=None, class_mapping=export / "class_mapping.json", expected_manifest_sha256=hashlib.sha256((export / "split_manifest.csv").read_bytes()).hexdigest(), expected_count=1))
            self.assertEqual(verified["status"], "PASS")
            atlas = build_candidate_review.build(argparse_namespace(data_root=export, manifest=export / "split_manifest.csv", output=root / "atlas", expected_count=1))
            self.assertEqual(atlas["candidate_polygons"], 1)
            self.assertIn("UNREVIEWED", (root / "atlas" / "review.csv").read_text(encoding="utf-8-sig"))
            self.assertTrue((root / "atlas" / "index.html").is_file())

    def test_missing_shapes_are_not_background(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotation = Path(temporary) / "empty.json"
            annotation.write_text(json.dumps({"imageWidth": 10, "imageHeight": 10, "shapes": []}), encoding="utf-8")
            labels, issues = verify_train_export.validate_annotation(annotation, 10, 10, {"Crack"})
            self.assertEqual(labels, [])
            self.assertTrue(any("background" in issue for issue in issues))


def argparse_namespace(**values):
    class Namespace:
        pass

    result = Namespace()
    for key, value in values.items():
        setattr(result, key, value)
    return result


if __name__ == "__main__":
    unittest.main()
