import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.exp14.aligned_support import A_FIELDS, B_FIELDS, cohort_rows, support_metrics
from tools.exp14.freeze_core import (
    FrozenSlotBatchSampler, build_slots, copy_verified, encode_labels, load_ultralytics_parser,
    make_schedule, training_protocol,
)
from tools.exp14.label_plan import polygon_rows


def candidate(variant, source, group):
    row = dict.fromkeys(A_FIELDS, "")
    row.update(variant_id=variant, source_id=source, group_id=group)
    return row


def image_record(source, kind):
    return {"source_id": source, "group_id": f"group-{source}", "split": "train",
            "image_path": f"images/{kind}/{source}.png", "label_path": f"labels/{kind}/{source}.txt",
            "image_sha256": f"{kind}-image", "label_sha256": f"{kind}-label",
            "instance_count": 1 if kind == "base" else 0}


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.a = [candidate("synthetic_V1", "synthetic", "g1"), candidate("other_V1", "other", "g2")]
        self.b = [{**self.a[0], **dict.fromkeys(B_FIELDS, "")}]
        self.details = {"synthetic_V1": {"a_status": "VERIFIED_CANDIDATE", "b_status": "VERIFIED_CANDIDATE"},
                        "other_V1": {"a_status": "VERIFIED_CANDIDATE", "b_status": "HOLD"}}

    def test_exact_b_subset_matches_a(self):
        a, b = cohort_rows(self.a, self.b, self.details, expected=(2, 1))
        self.assertEqual(a.keys() - b.keys(), {"other_V1"})

    def test_missing_edited_or_unreviewed_cohort_fails(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            cohort_rows(self.a, [], self.details, expected=(2, 1))
        with self.assertRaisesRegex(ValueError, "identity"):
            cohort_rows(self.a, [dict(self.b[0], final_image_sha256="changed")], self.details, expected=(2, 1))
        self.details["synthetic_V1"]["b_status"] = "PENDING_HUMAN_REVIEW"
        with self.assertRaisesRegex(ValueError, "candidate sets"):
            cohort_rows(self.a, self.b, self.details, expected=(2, 1))

    def test_duplicate_group_fails(self):
        self.a[1]["group_id"] = "g1"
        with self.assertRaisesRegex(ValueError, "Duplicate or empty group_id"):
            cohort_rows(self.a, self.b, self.details, expected=(2, 1))


class SupportTests(unittest.TestCase):
    def test_separates_native_black_from_visible_black_fill_and_target_loss(self):
        original = Image.new("RGB", (6, 6), (90, 90, 90))
        aligned = original.copy()
        valid = Image.new("L", (6, 6), 255)
        original.putpixel((0, 0), (0, 0, 0))
        for xy in ((0, 0), (1, 0)):
            aligned.putpixel(xy, (0, 0, 0))
            valid.putpixel(xy, 0)
        aligned.putpixel((5, 5), (0, 0, 0))
        triangle = {"points": [[1, 0], [4, 0], [4, 4]]}
        report, _, _ = support_metrics(original, aligned, valid, {"synthetic": triangle})
        self.assertEqual(report["invalid_pixels"], 2)
        self.assertEqual(report["exact_black_outside_support_pixels"], 2)
        self.assertEqual(report["fill_over_visible_source_pixels_luma_ge25"], 1)
        self.assertEqual(report["fill_over_dark_source_pixels_luma_lt25"], 1)
        self.assertEqual(report["regions"]["synthetic"]["target_pixels"] -
                         report["regions"]["synthetic"]["valid_pixels"], 1)
        self.assertEqual(sum(map(sum, report["invalid_tiles_3x3"])), 2)

    def test_invalid_masks_or_near_black_do_not_count_as_exact_black(self):
        original = Image.new("RGB", (6, 6), (90, 90, 90))
        aligned = Image.new("RGB", (6, 6), (1, 0, 0))
        valid = Image.new("L", (6, 6), 0)
        report, _, _ = support_metrics(original, aligned, valid, {})
        self.assertEqual(report["exact_black_outside_support_pixels"], 0)
        valid.putpixel((0, 0), 127)
        with self.assertRaisesRegex(ValueError, "mask"):
            support_metrics(original, aligned, valid, {})


class FreezeTests(unittest.TestCase):
    def test_exact_baseline_format_and_explicit_empty_file_bytes(self):
        shape = {"shape_type": "polygon", "label": "Crack", "points": [[0, 0], [4, 0], [4, 4]]}
        labels = polygon_rows("synthetic", [shape], (8, 8))
        self.assertEqual(encode_labels(labels),
                         b"1 0.0000000000 0.0000000000 0.5000000000 0.0000000000 0.5000000000 0.5000000000\n")
        self.assertEqual(encode_labels({}), b"")

    def test_file_copy_is_hash_bound_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            target = Path(directory) / "frozen"
            source.write_bytes(b"synthetic")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "changed"):
                copy_verified(source, target, "0" * 64)
            copy_verified(source, target, digest)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            with self.assertRaises(FileExistsError):
                copy_verified(source, target, digest)

    def test_all_slots_match_and_e0_repeats_index_not_path_in_dataset(self):
        base = {sid: image_record(sid, "base") for sid in ("one", "two")}
        derived = {"one": image_record("one", "aligned")}
        slots = build_slots(base, derived)
        schedule = make_schedule(slots, epochs=2, seed=42, batch=2)
        self.assertEqual(schedule, make_schedule(slots, epochs=2, seed=42, batch=2))
        self.assertEqual([len(b) for b in schedule[0]["batches"]], [2, 1])
        e0 = FrozenSlotBatchSampler(slots, schedule, ["images/base/one.png", "images/base/two.png"], "e0")
        e1 = FrozenSlotBatchSampler(slots, schedule,
                                   ["images/base/one.png", "images/base/two.png", "images/aligned/one.png"], "e1")
        for epoch in (1, 2):
            e0.set_epoch(epoch)
            e1.set_epoch(epoch)
            self.assertEqual(sorted(i for batch in e0 for i in batch), [0, 0, 1])
            self.assertEqual(sorted(i for batch in e1 for i in batch), [0, 1, 2])

    def test_sampler_rejects_extra_paths_and_duplicate_epoch_slots(self):
        base = {"one": image_record("one", "base")}
        with self.assertRaisesRegex(ValueError, "not from baseline"):
            build_slots(base, {"two": image_record("two", "aligned")})
        slots = build_slots(base, {})
        schedule = make_schedule(slots, epochs=1)
        with self.assertRaisesRegex(ValueError, "image set"):
            FrozenSlotBatchSampler(slots, schedule, ["images/base/one.png", "extra.png"], "e0")
        schedule[0]["batches"][0].append("base:one")
        with self.assertRaisesRegex(ValueError, "slot set"):
            FrozenSlotBatchSampler(slots, schedule, ["images/base/one.png"], "e0")

    def test_training_protocol_disables_secondary_sampling(self):
        protocol = training_protocol("frozen", 2200)
        self.assertTrue(all(protocol[k] == 0 for k in ("mosaic", "mixup", "cutmix", "copy_paste")))
        self.assertFalse(protocol["drop_last"])
        self.assertEqual(protocol["accumulate"], 1)
        self.assertEqual(protocol["optimizer_updates_each_arm"], 2200)


@unittest.skipUnless(os.environ.get("EXP14_TEST_PARSER_WHEEL"), "Pinned parser wheel optional for synthetic test")
class PinnedParserTest(unittest.TestCase):
    def test_empty_segmentation_label_is_found_and_not_missing(self):
        img2label, parser, _ = load_ultralytics_parser(Path(os.environ["EXP14_TEST_PARSER_WHEEL"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            image = root / "images" / "synthetic.png"
            label = root / "labels" / "synthetic.txt"
            Image.new("RGB", (24, 24)).save(image)
            label.write_bytes(b"")
            self.assertEqual(img2label([str(image)]), [str(label)])
            parsed = parser((str(image), str(label), "", False, 7, 0, 2, False))
            self.assertEqual(parsed[1].shape, (0, 5))
            self.assertEqual(parsed[3], [])
            self.assertEqual(parsed[5:9], (0, 1, 1, 0))


if __name__ == "__main__":
    unittest.main()
