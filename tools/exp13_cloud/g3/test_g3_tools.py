import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from tools.exp13_cloud.g2.audit_g2_closeout import summarize_sources
from tools.exp13_cloud.g3 import build_candidate_review, verify_train_export


def args(**values):
    return argparse.Namespace(**values)


class G3FormalPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.train_count = 3
        self.train_instances = 5
        self.rows = []
        self.make_frozen_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def make_frozen_manifest(self, include_val=False):
        self.rows = []
        fields = ["image_path", "json_path", "stem", "split", "group_id", "sha256", "labels_present", "instance_count", "near_duplicate_group_id", "source_status", "annotation_version", "image_suffix", "image_width", "image_height", "json_sha256", "label_path", "label_sha256"]
        for index, labels in enumerate((["Crack", "Burn"], ["corrosion"], ["Crack", "Crack"]), 1):
            stem = str(index)
            image_rel, json_rel = f"images/{stem}.png", f"annotations/{stem}.json"
            image_path, json_path = self.raw / image_rel, self.raw / json_rel
            image_path.parent.mkdir(parents=True, exist_ok=True); json_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), (80 + index, 100, 120)).save(image_path)
            shapes = [{"label": label, "shape_type": "polygon", "points": [[2, 2], [12, 2], [8, 12]]} for label in labels]
            json_path.write_text(json.dumps({"imagePath": f"{stem}.png", "imageWidth": 32, "imageHeight": 24, "version": "fixture-v1", "shapes": shapes}), encoding="utf-8")
            self.rows.append({"image_path": image_rel, "json_path": json_rel, "stem": stem, "split": "train", "group_id": f"group_{stem}", "sha256": verify_train_export.sha256_file(image_path), "labels_present": "|".join(sorted(set(labels))), "instance_count": str(len(labels)), "near_duplicate_group_id": "", "source_status": "matched_authoritative", "annotation_version": "fixture-v1", "image_suffix": ".png", "image_width": "32", "image_height": "24", "json_sha256": verify_train_export.sha256_file(json_path), "label_path": "", "label_sha256": ""})
        if include_val:
            self.rows.append({"image_path": "images/val-never-open.png", "json_path": "annotations/val-never-open.json", "stem": "999", "split": "val", "group_id": "group_val", "sha256": "UNKNOWN", "labels_present": "Crack", "instance_count": "1", "near_duplicate_group_id": "", "source_status": "matched_authoritative", "annotation_version": "fixture-v1", "image_suffix": ".png", "image_width": "32", "image_height": "24", "json_sha256": "UNKNOWN", "label_path": "", "label_sha256": ""})
        self.manifest = self.root / "frozen_split_manifest.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(self.rows)
        self.expected_hash = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.mapping = self.root / "class_mapping.json"
        self.mapping.write_text(json.dumps({"names": verify_train_export.CLASSES}), encoding="utf-8")

    def run_export(self, output):
        return verify_train_export.export_train(args(mode="export", manifest=self.manifest, source_root=self.raw, output=output, class_mapping=self.mapping, expected_manifest_sha256=self.expected_hash, expected_count=self.train_count, expected_instances=self.train_instances))

    def verify(self, root, report_path=None):
        return verify_train_export.validate_export_data(root / "frozen_split_manifest.csv", root / "export_manifest.csv", root, root / "class_mapping.json", self.expected_hash, self.train_count, self.train_instances)

    def test_frozen_to_export_archive_windows_verify_and_atlas(self):
        export = self.root / "export"
        record = self.run_export(export)
        self.assertEqual(record["train_images"], 3)
        self.assertEqual(record["train_instances"], 5)
        self.assertEqual(record["class_instance_counts"], {"Burn": 1, "Crack": 3, "corrosion": 1})
        with zipfile.ZipFile(export / "train_export.zip") as archive:
            unpacked = self.root / "unpacked"
            archive.extractall(unpacked)
        self.assertEqual((unpacked / "frozen_split_manifest.csv").read_bytes(), self.manifest.read_bytes())
        report_path = self.root / "verification.json"
        report = verify_train_export.verify(args(manifest=unpacked / "frozen_split_manifest.csv", export_manifest=unpacked / "export_manifest.csv", source_root=unpacked, class_mapping=unpacked / "class_mapping.json", output=report_path, archive=export / "train_export.zip", expected_manifest_sha256=self.expected_hash, expected_count=self.train_count, expected_instances=self.train_instances))
        self.assertEqual(report["status"], "PASS", report["issues"])
        atlas = self.root / "atlas"
        result = build_candidate_review.build(args(data_root=unpacked, manifest=unpacked / "frozen_split_manifest.csv", export_manifest=unpacked / "export_manifest.csv", verification_report=report_path, class_mapping=unpacked / "class_mapping.json", output=atlas, expected_manifest_sha256=self.expected_hash, expected_count=self.train_count, expected_instances=self.train_instances))
        self.assertEqual(result["train_images"], 3)
        self.assertEqual(result["candidate_polygons"], 5)
        self.assertEqual(result["independent_source_count"], 3)
        self.assertEqual((result["class_source_counts"]["Crack"]), 2)
        with (atlas / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(row["review_status"] == "UNREVIEWED" for row in rows))
        self.assertIn("loading=\"lazy\"", (atlas / "index.html").read_text(encoding="utf-8"))

    def test_changed_frozen_manifest_rejected(self):
        export = self.root / "export"; self.run_export(export)
        with (export / "frozen_split_manifest.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        self.assertEqual(self.verify(export)["status"], "FAIL")

    def test_export_manifest_mutation_rejected(self):
        export = self.root / "export"; self.run_export(export)
        text = (export / "export_manifest.csv").read_text(encoding="utf-8")
        (export / "export_manifest.csv").write_text(text.replace("group_1", "wrong_group", 1), encoding="utf-8")
        self.assertEqual(self.verify(export)["status"], "FAIL")

    def test_missing_json_rejected(self):
        export = self.root / "export"; self.run_export(export)
        (export / "annotations/train/2.json").unlink()
        self.assertEqual(self.verify(export)["status"], "FAIL")

    def test_image_hash_error_rejected(self):
        export = self.root / "export"; self.run_export(export)
        Image.new("RGB", (32, 24), "red").save(export / "images/train/2.png")
        self.assertEqual(self.verify(export)["status"], "FAIL")

    def test_wrong_source_set_with_same_count_rejected(self):
        export = self.root / "export"; self.run_export(export)
        path = export / "export_manifest.csv"
        rows = verify_train_export.read_rows(path)
        rows[-1]["source_id"] = "999"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=verify_train_export.EXPORT_COLUMNS); writer.writeheader(); writer.writerows(rows)
        self.assertEqual(self.verify(export)["status"], "FAIL")

    def test_non_train_extra_file_rejected(self):
        export = self.root / "export"; self.run_export(export)
        Image.new("RGB", (1, 1)).save(export / "images/train/999.png")
        self.assertEqual(self.verify(export)["status"], "FAIL")

    def test_archive_member_bytes_must_match_unpacked_files(self):
        export = self.root / "export"; self.run_export(export)
        archive_path = export / "train_export.zip"
        replacement = self.root / "changed.zip"
        with zipfile.ZipFile(archive_path) as original, zipfile.ZipFile(replacement, "w", zipfile.ZIP_DEFLATED) as changed:
            for name in original.namelist():
                content = original.read(name)
                if name == "images/train/1.png":
                    content += b"tampered"
                changed.writestr(name, content)
        replacement.with_suffix(".zip.sha256").write_text(
            f"{verify_train_export.sha256_file(replacement)}  {replacement.name}\n", encoding="ascii")
        unpacked = self.root / "unpacked"
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(unpacked)
        report = {"export_manifest": str(unpacked / "export_manifest.csv")}
        issues = verify_train_export.verify_archive(replacement, report, unpacked)
        self.assertEqual(issues[0]["type"], "archive_content")

    def test_atlas_rejects_failed_or_missing_gate(self):
        export = self.root / "export"; self.run_export(export)
        output = self.root / "atlas_missing_gate"
        with self.assertRaises(RuntimeError):
            build_candidate_review.build(args(data_root=export, manifest=export / "frozen_split_manifest.csv", export_manifest=export / "export_manifest.csv", verification_report=self.root / "missing.json", class_mapping=export / "class_mapping.json", output=output, expected_manifest_sha256=self.expected_hash, expected_count=self.train_count, expected_instances=self.train_instances))
        failed = self.root / "failed.json"; failed.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            build_candidate_review.build(args(data_root=export, manifest=export / "frozen_split_manifest.csv", export_manifest=export / "export_manifest.csv", verification_report=failed, class_mapping=export / "class_mapping.json", output=output, expected_manifest_sha256=self.expected_hash, expected_count=self.train_count, expected_instances=self.train_instances))

    def test_val_metadata_only_is_not_opened(self):
        self.make_frozen_manifest(include_val=True)
        # Export reads TRAIN pairs only; deliberately absent VAL files are metadata-only.
        export = self.root / "export_val_meta"
        self.run_export(export)
        self.assertTrue((export / "export_manifest.csv").is_file())

    def test_g2_single_source_error_forces_top_level_failure(self):
        sources = {"260": {"execution_state": "EXECUTION_VERIFIED", "execution": {"billing_state": "COST_UNKNOWN", "upstream_identity_state": "UPSTREAM_IDENTITY_UNKNOWN"}, "issues": ["260: hash mismatch"], "geometry_state": "GEOMETRY_CANDIDATE"}, "470": {"execution_state": "EXECUTION_VERIFIED", "execution": {"billing_state": "COST_UNKNOWN", "upstream_identity_state": "UPSTREAM_IDENTITY_UNKNOWN"}, "issues": [], "geometry_state": "GEOMETRY_UNVERIFIED"}}
        summary = summarize_sources(sources)
        self.assertEqual(summary["status"], "FAIL")
        self.assertIn("260", summary["source_errors"])
        self.assertEqual(summary["billing"]["state"], "COST_UNKNOWN")
        self.assertEqual(summary["upstream_identity"]["state"], "UPSTREAM_IDENTITY_UNKNOWN")
        self.assertFalse(summary["trusted_pair_pass"])


if __name__ == "__main__":
    unittest.main()
