"""Test Exp14-A10 train membership and materialized label safeguards."""

import hashlib
import unittest
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from PIL import Image

from tools.exp14.a10_dataset_entries import (
    ARM_CONFIGS, CLASSES, EMPTY_SHA256, EVAL_ROOT, arm_counts, bind_entry, config_for,
    verify_inventory, verify_local_parser, verify_paths, write_configs,
)


class TestDatasetEntries(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.dataset = self.root / "dataset"
        self.configs = self.root / "configs"
        self.eval = PurePosixPath(EVAL_ROOT)
        self.base = self.make_entry("base", "34", b"1 .1 .1 .5 .1 .2 .5\n", (35, 98, 15))
        self.derived = self.make_entry("pseudo_normal", "34", b"", (98, 35, 15))

    def make_entry(self, folder, stem, label_bytes, color):
        ir = f"images/{folder}/{stem}.png"
        lr = f"labels/{folder}/{stem}.txt"
        ip, lp = self.dataset / ir, self.dataset / lr
        ip.parent.mkdir(parents=True, exist_ok=True)
        lp.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(ip)
        lp.write_bytes(label_bytes)
        return {"source_id": stem, "group_id": "g", "split": "train",
                "image_path": ir, "label_path": lr,
                "image_sha256": hashlib.sha256(ip.read_bytes()).hexdigest(),
                "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
                "instance_count": int(bool(label_bytes))}

    def frozen_row(self):
        return {**self.derived, "published_path": self.dataset / self.derived["image_path"],
                "published_sha256": self.derived["image_sha256"],
                "label_path": self.dataset / self.derived["label_path"]}

    def test_a10_asset_cannot_cross_its_group_or_held_out_bytes(self):
        row = self.frozen_row()
        normalized = bind_entry(self.root, self.dataset, row, self.base, set())
        self.assertEqual(normalized, self.derived)
        self.assertEqual(normalized["label_sha256"], EMPTY_SHA256)
        with self.assertRaisesRegex(ValueError, "original frozen TRAIN group"):
            bind_entry(self.root, self.dataset, {**row, "group_id": "bad"}, self.base, set())
        with self.assertRaisesRegex(ValueError, "held-out image"):
            bind_entry(self.root, self.dataset, row, self.base, {row["published_sha256"]})

    def test_e0_exact_e1_union_and_common_original_references(self):
        inventory = {"baseline": [self.base], "derived": [self.derived]}
        write_configs(self.configs, inventory["baseline"], inventory["derived"], self.eval)
        verify_paths(self.configs, self.root, inventory, self.eval)
        self.assertEqual(config_for("e0", self.eval)["names"], CLASSES)
        for split in ("val", "test"):
            self.assertEqual(config_for("e0", self.eval)[split], config_for("e1", self.eval)[split])
        self.assertEqual(arm_counts(inventory["baseline"]), {"train_images": 1, "train_labels": 1,
                         "original_images": 1, "explicit_empty_labels": 0, "segmentation_instances": 1})
        self.assertEqual(arm_counts([self.base, self.derived])["explicit_empty_labels"], 1)
        list_path = self.configs / ARM_CONFIGS["e0"][0]
        image = list_path.read_text(encoding="utf-8")
        list_path.write_text(image * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "TRAIN entry list differs"):
            verify_paths(self.configs, self.root, inventory, self.eval)

    def test_parser_recognizes_existing_empty_labels_and_rejects_stale_cache(self):
        count = verify_local_parser(self.dataset, [self.base, self.derived], None)
        self.assertEqual(count["empty_labels"], 1)
        self.assertEqual(count["positive_labels"], 1)
        self.assertEqual(count["segmentation_instances"], 1)
        missing = self.dataset / self.derived["label_path"]
        missing.rename(missing.with_suffix(".tmp"))
        with self.assertRaises(FileNotFoundError):
            verify_local_parser(self.dataset, [self.base, self.derived], None)
        (self.dataset / "labels" / "base.cache").touch()
        with self.assertRaisesRegex(ValueError, "Unexpected dataset cache"):
            verify_local_parser(self.dataset, [self.base, self.derived], None)

    def test_inventory_replay_enforces_frozen_baseline_identity(self):
        inventory = {"baseline": [self.base], "derived": [self.derived]}
        frozen = {**self.base, "sha256": self.base["image_sha256"]}
        sealed = {"train": {"34": frozen}, "val": {}, "test": {}}
        verify_inventory(inventory, sealed)
        inventory["baseline"] = [{**self.base, "label_sha256": EMPTY_SHA256}]
        with self.assertRaisesRegex(ValueError, "Original TRAIN image or label identity differs"):
            verify_inventory(inventory, sealed)


if __name__ == "__main__":
    unittest.main()
