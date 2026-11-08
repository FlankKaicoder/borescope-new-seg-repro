import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("build_g3c0_review_page.py")
SPEC = importlib.util.spec_from_file_location("build_g3c0_review_page", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReviewPageBuilderTests(unittest.TestCase):
    def setUp(self):
        self.expected = {f"s{i:03d}": "GEOMETRY_CANDIDATE" if i < 28 else "ALIGNMENT_UNVERIFIED"
                         for i in range(36)}

    def write_csv(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MODULE.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def rows(self):
        return [{"source_id": sid, "visual_review_status": "VISUAL_PASS",
                 "geometry_status": geometry, "review_notes": "中文, \"备注\"\n第二行" if i == 0 else ""}
                for i, (sid, geometry) in enumerate(self.expected.items())]

    def test_recovery_csv_exact_set_and_multiline_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recovery.txt"
            self.write_csv(path, self.rows())
            rows = MODULE.parse_csv(path, self.expected)
            self.assertEqual(len(rows), 36)
            self.assertEqual(rows[0]["review_notes"], "中文, \"备注\"\n第二行")

    def test_reject_duplicate_missing_invalid_visual_and_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recovery.csv"
            rows = self.rows()
            for invalid in (rows[:-1] + [rows[0]],
                            rows[:-1],
                            [{**r, "visual_review_status": "APPROVE"} if i == 0 else r for i, r in enumerate(rows)],
                            [{**r, "geometry_status": "ALIGNMENT_UNVERIFIED"} if i == 0 else r for i, r in enumerate(rows)]):
                self.write_csv(path, invalid)
                with self.assertRaises(ValueError):
                    MODULE.parse_csv(path, self.expected)

    def test_enhanced_page_keeps_local_images_and_embeds_recovery(self):
        cards = []
        rows = self.rows()
        for row in rows:
            cards.append(f'<article data-class="Burn" data-source="{row["source_id"]}"><label>Geometry <select data-field="geometry"><option>{row["geometry_status"]}</option></select></label></article>')
        html = '<button id="download">Download CSV</button><span id="count"></span><main>' + ''.join(cards) + '</main><script>old()</script>'
        with tempfile.TemporaryDirectory() as temp:
            js = Path(temp) / "ui.js"
            js.write_text("const state = __INITIAL_REVIEW__;", encoding="utf-8")
            result, info = MODULE.enhance_html(html, js, rows)
        self.assertEqual(info["source_count"], 36)
        self.assertIn("Import Review CSV", result)
        self.assertIn("const state = ", result)
        self.assertIn("中文, \\\"备注\\\"\\n第二行", result)


if __name__ == "__main__":
    unittest.main()
