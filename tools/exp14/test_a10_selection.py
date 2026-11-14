import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.exp14 import a10_selection as selection


class SelectionImportTest(unittest.TestCase):
    def setUp(self):
        self.source = "existing_source"
        self.primary = self.make_asset("a", "b", "ALIGNED")
        self.alternate = self.make_asset("c", "d", "RAW")
        self.assets = {x["asset_id"]: x for x in (self.primary, self.alternate)}
        self.rows = {self.source: {"recommended_asset_id": self.primary["asset_id"], "disposition": selection.PROPOSED}}

    def make_asset(self, review_symbol, hash_symbol, asset_type):
        rid, digest = review_symbol * 64, hash_symbol * 64
        return {"asset_type": asset_type, "review_id": rid, "variant_id": "historical_variant", "sha256": digest,
                "asset_id": selection.asset_id(rid, asset_type, digest)}

    def choice(self, asset, action):
        return {"batch_id": selection.BATCH_ID, "exact_set_sha256": selection.EXACT_SET_SHA256,
                "source_id": self.source, "action": action, "asset_id": asset["asset_id"],
                "review_id": asset["review_id"], "variant_id": asset["variant_id"], "asset_type": asset["asset_type"],
                "asset_sha256": asset["sha256"], "recorded_utc": "2026-11-14T10:30:00.042Z",
                "operation_id": "", "batch_scope": "", "reason": "", "notes": ""}

    def test_accept_and_explicit_alternate_are_distinct(self):
        accepted = self.choice(self.primary, selection.ACCEPT)
        changed = self.choice(self.alternate, selection.ALTERNATE)
        self.assertIs(selection.validate_choice(accepted, self.rows, self.assets), self.primary)
        self.assertIs(selection.validate_choice(changed, self.rows, self.assets), self.alternate)
        changed["action"] = selection.ACCEPT
        with self.assertRaisesRegex(ValueError, "recommendation mismatch"):
            selection.validate_choice(changed, self.rows, self.assets)
        self.rows[self.source]["recommended_asset_id"] = None
        changed["action"] = selection.ALTERNATE
        self.assertIs(selection.validate_choice(changed, self.rows, self.assets), self.alternate)

    def test_wrong_group_asset_batch_time_and_scope_are_rejected(self):
        accepted = self.choice(self.primary, selection.ACCEPT)
        for key, value, detail in (
            ("batch_id", "other-batch", "different review batch"),
            ("asset_sha256", "f" * 64, "Malformed selected asset identity"),
            ("variant_id", "different_variant", "tuple mismatch"),
            ("recorded_utc", "2026-11-14T10:30:00", "not a UTC timestamp"),
            ("batch_scope", self.source, "batch scope invalid"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, detail):
                selection.validate_choice({**accepted, key: value}, self.rows, self.assets)
        bulk = {**accepted, "operation_id": "operation", "batch_scope": self.source}
        self.assertIs(selection.validate_choice(bulk, self.rows, self.assets), self.primary)
        bulk["batch_scope"] += "|untracked_source"
        with self.assertRaisesRegex(ValueError, "batch scope invalid"):
            selection.validate_choice(bulk, self.rows, self.assets)

    def test_only_exact_asset_note_is_exposed(self):
        visual = {"review_id": self.primary["review_id"], "asset_observations": {"RAW": "another image", "ALIGNED": "same image"},
                  "alternate_observations": {self.alternate["review_id"]: {"RAW": "alternate image"}}}
        self.assertEqual(selection.exact_visual_note(visual, self.primary), "same image")
        self.assertEqual(selection.exact_visual_note(visual, self.alternate), "alternate image")
        visual["alternate_observations"] = {}
        self.assertEqual(selection.exact_visual_note(visual, self.alternate), "")

    def test_csv_parser_preserves_authoritative_raw_input_and_restores_multiline_notes(self):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=selection.FIELDS)
        writer.writeheader()
        writer.writerow({**self.choice(self.primary, selection.ACCEPT), "notes": "'=第一行,带引号\"\n第二行"})
        data = buffer.getvalue().encode("utf-8-sig")
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
            selection, SELECTION_BYTES=len(data), SELECTION_SHA256=hashlib.sha256(data).hexdigest()
        ):
            file = Path(tmp) / "selection.csv"
            file.write_bytes(data)
            raw, rows = selection.read_selection(file)
            self.assertEqual(raw, data)
            self.assertEqual(rows[0]["notes"], '=第一行,带引号"\n第二行')
            file.write_bytes(data.replace(b"2026", b"2025"))
            with self.assertRaisesRegex(ValueError, "original bytes"):
                selection.read_selection(file)

    def test_inline_bundle_is_read_as_json_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "index.html"
            html = '<script>throw Error("not executed")</script><script id="bundle" type="application/json">{}'
            page.write_text(html.format(json.dumps({"rows": [1]})) + "</script>", encoding="utf-8")
            self.assertEqual(selection.inline_bundle(page), {"rows": [1]})


if __name__ == "__main__":
    unittest.main()
