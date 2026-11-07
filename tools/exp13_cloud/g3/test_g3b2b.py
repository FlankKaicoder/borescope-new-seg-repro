from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from tools.exp13_cloud.g3.g3_batch_executor import load_protocol, run_batch
from tools.exp13_cloud.g3.prepare_g3b2b_batch import build_review_page, choose_corrosion


REPO = Path(__file__).resolve().parents[3]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


class MockTransport:
    def __init__(self, responses: list[tuple[int, dict[str, str], bytes]]):
        self.responses = list(responses)
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls = 0

    def post(self, endpoint: str, headers: dict[str, str], body: bytes):
        self.post_calls.append({"endpoint": endpoint, "headers": headers, "body": body})
        if not self.responses:
            raise AssertionError("Unexpected mock POST")
        return self.responses.pop(0)

    def get(self, url: str):
        self.get_calls += 1
        raise AssertionError("Unexpected mock GET")


class G3B2BExecutorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="g3b2b_synthetic_")
        self.root = Path(self.temp.name)
        self.private = self.root / "private"
        self.batch = self.private / "batch"
        self.exec_root = self.private / "execution"
        (self.batch / "inputs").mkdir(parents=True)
        (self.batch / "masks").mkdir()

        image = Image.new("RGB", (8, 6), (70, 80, 90))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        self.image_bytes = image_bytes.getvalue()
        (self.batch / "inputs/source_a.png").write_bytes(self.image_bytes)

        mask = Image.new("RGBA", (8, 6), (255, 255, 255, 255))
        mask.putpixel((2, 2), (255, 255, 255, 0))
        mask_bytes = io.BytesIO()
        mask.save(mask_bytes, format="PNG")
        self.mask_bytes = mask_bytes.getvalue()
        (self.batch / "masks/source_a.png").write_bytes(self.mask_bytes)

        self.prompt_path = Path("configs/exp13_cloud/g2/CRACK_RESTORATION_V1.txt")
        prompt_bytes = (REPO / self.prompt_path).read_bytes()
        prompt_wire = prompt_bytes[:-1]
        self.prompt_manifest = {"record_type": "SYNTHETIC_FIXTURE", "prompts": {"Crack": {
            "prompt_id": "CRACK_RESTORATION_V1", "path": self.prompt_path.as_posix(),
            "file_sha256": sha(prompt_bytes), "wire_sha256": sha(prompt_wire),
        }}}
        (self.batch / "prompt_manifest.json").write_text(json.dumps(self.prompt_manifest), encoding="utf-8")

        self.manifest = self.batch / "expanded_batch_manifest.csv"
        self.review = self.batch / "review.csv"
        fields = ["source_id", "split", "class", "decision", "priority", "selected_polygon_ids",
                  "source_image_sha256", "source_json_sha256", "image_width", "image_height", "image_path",
                  "batch_image_path", "mask_path", "mask_sha256", "prompt_id", "prompt_file",
                  "prompt_file_sha256", "prompt_wire_sha256", "status", "manually_approved",
                  "authorized_for_paid_post", "request_count_proposed", "roi_policy", "n"]
        self.rows = []
        for sid in ("source_a", "source_b"):
            row = {
                "source_id": sid, "split": "train", "class": "Crack", "decision": "SELECT", "priority": "A",
                "selected_polygon_ids": sid + "_000", "source_image_sha256": sha(self.image_bytes),
                "source_json_sha256": "a" * 64, "image_width": "8", "image_height": "6",
                "image_path": "images/train/" + sid + ".png", "batch_image_path": "inputs/source_a.png",
                "mask_path": "masks/source_a.png", "mask_sha256": sha(self.mask_bytes), "prompt_id": "CRACK_RESTORATION_V1",
                "prompt_file": self.prompt_path.as_posix(), "prompt_file_sha256": sha(prompt_bytes),
                "prompt_wire_sha256": sha(prompt_wire), "status": "PROPOSED", "manually_approved": "NO",
                "authorized_for_paid_post": "NO", "request_count_proposed": "1",
                "roi_policy": "FULL_NATIVE_FRAME_NO_CROP_RESIZE_PADDING_OR_COORDINATE_TRANSFORM", "n": "1",
            }
            self.rows.append(row)
        write_csv(self.manifest, self.rows, fields)
        self.review_rows = [{
            "source_id": row["source_id"], "source_image_sha256": row["source_image_sha256"],
            "prompt_id": row["prompt_id"], "prompt_file_sha256": row["prompt_file_sha256"],
            "review_status": "APPROVE", "review_notes": "synthetic fixture",
        } for row in self.rows]
        review_fields = ["source_id", "source_image_sha256", "prompt_id", "prompt_file_sha256", "review_status", "review_notes"]
        write_csv(self.review, self.review_rows, review_fields)

        protocol, protocol_sha = load_protocol(REPO)
        (self.batch / "prompt_manifest.json").write_bytes(json.dumps(self.prompt_manifest, sort_keys=True).encode())
        self.auth = self.private / "synthetic_authorization.json"
        self.auth_value = {
            "record_type": "G3B2B_LIVE_AUTHORIZATION", "status": "G3B2B_BATCH_LIVE_APPROVED",
            "independent_gate_pass": True, "user_approved": True,
            "batch_manifest_sha256": sha(self.manifest.read_bytes()), "review_csv_sha256": sha(self.review.read_bytes()),
            "prompt_manifest_sha256": sha((self.batch / "prompt_manifest.json").read_bytes()),
            "protocol_sha256": protocol_sha, "approved_source_ids": ["source_a", "source_b"],
            "request_count_cap": 2, "provider": protocol["request"]["base_url"], "model": protocol["request"]["model"],
            "endpoint": protocol["request"]["endpoint"], "n": 1, "unit_price_cap": 0.5,
            "total_spend_cap": 1.0, "currency": "USD", "price_evidence_sha256": "b" * 64,
            "provider_hard_limit_evidence": "synthetic fixture only", "supported_input_resolutions": [[8, 6]],
            "approval_reference": "SYNTHETIC_TEST_ONLY",
        }
        self.auth.write_text(json.dumps(self.auth_value), encoding="utf-8")
        output_image = Image.new("RGB", (7, 5), (90, 80, 70))
        output_bytes = io.BytesIO()
        output_image.save(output_bytes, format="PNG")
        self.output_bytes = output_bytes.getvalue()
        self.success = (200, {"content-type": "application/json", "x-request-id": "mock"}, json.dumps({
            "model": "mock-model", "data": [{"b64_json": base64.b64encode(self.output_bytes).decode()}],
            "usage": {"cost": 0.25, "total_tokens": 2},
        }).encode())

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, live: bool, transport: MockTransport | None = None, auth_path: Path | None = None, api_key: Any = "fake-key"):
        return run_batch(self.manifest, self.review, self.batch, self.exec_root, REPO, self.private,
                         live=live, auth_path=auth_path or self.auth, api_key=api_key, transport=transport)

    def test_dry_run_builds_two_requests_without_post(self):
        fake = MockTransport([])
        summary = self.execute(False, fake)
        self.assertEqual(summary["api_post_count"], 0)
        self.assertEqual(len(fake.post_calls), 0)
        self.assertEqual(len(list((self.exec_root / "dry_run").glob("*/request_preview.json"))), 2)

    def test_prompt_manifest_must_match_source_prompt_identity(self):
        prompt = json.loads((self.batch / "prompt_manifest.json").read_text())
        prompt["prompts"]["Crack"]["file_sha256"] = "0" * 64
        (self.batch / "prompt_manifest.json").write_text(json.dumps(prompt), encoding="utf-8")
        fake = MockTransport([])
        with self.assertRaisesRegex(ValueError, "differs from batch Prompt manifest"):
            self.execute(False, fake)
        self.assertEqual(len(fake.post_calls), 0)

    def test_mock_live_saves_outputs_and_restart_never_posts_again(self):
        fake = MockTransport([self.success, self.success])
        first = self.execute(True, fake)
        self.assertEqual(first["api_post_count"], 2)
        self.assertEqual(len(fake.post_calls), 2)
        source_output = self.exec_root / "live/source_a"
        self.assertEqual(sha((source_output / "raw_output.bin").read_bytes()), sha(self.output_bytes))
        self.assertEqual(json.loads((source_output / "TERMINAL.json").read_text())["post_invocations"], 1)
        second = self.execute(True, fake)
        self.assertEqual(second["api_post_count"], 0)
        self.assertTrue(all(item["status"] == "ALREADY_SUCCEEDED_NO_POST" for item in second["results"]))
        self.assertEqual(len(fake.post_calls), 2)

    def test_http_failure_stops_and_permanent_lock_blocks_restart(self):
        failure = (503, {"content-type": "application/json"}, b'{"error":"mock"}')
        fake = MockTransport([failure])
        with self.assertRaisesRegex(RuntimeError, "no retry"):
            self.execute(True, fake)
        self.assertEqual(len(fake.post_calls), 1)
        self.assertTrue((self.exec_root / "locks/source_a.json").is_file())
        self.assertFalse((self.exec_root / "locks/source_b.json").exists())
        with self.assertRaisesRegex(RuntimeError, "non-successfully"):
            self.execute(True, fake)
        self.assertEqual(len(fake.post_calls), 1)

    def test_existing_output_is_never_overwritten_or_posted(self):
        existing = self.exec_root / "live/source_a"
        existing.mkdir(parents=True)
        sentinel = existing / "keep.txt"
        sentinel.write_text("unchanged")
        fake = MockTransport([self.success])
        with self.assertRaisesRegex(RuntimeError, "without request lock"):
            self.execute(True, fake)
        self.assertEqual(len(fake.post_calls), 0)
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_authorization_exact_set_and_resolution_are_checked_before_key_or_post(self):
        bad_auth = dict(self.auth_value, approved_source_ids=["source_a"], request_count_cap=1)
        bad_path = self.private / "bad_auth.json"
        bad_path.write_text(json.dumps(bad_auth))
        key_reads = []
        fake = MockTransport([])
        with self.assertRaisesRegex(ValueError, "exact-set"):
            self.execute(True, fake, bad_path, lambda: key_reads.append(True))
        self.assertEqual(key_reads, [])
        self.assertEqual(len(fake.post_calls), 0)

        bad_auth = dict(self.auth_value, supported_input_resolutions=[[800, 600]])
        bad_path.write_text(json.dumps(bad_auth))
        with self.assertRaisesRegex(ValueError, "resolution evidence"):
            self.execute(True, fake, bad_path, lambda: key_reads.append(True))
        self.assertEqual(key_reads, [])
        self.assertEqual(len(fake.post_calls), 0)

    def test_missing_reported_cost_stops_after_one_post_and_blocks_resume(self):
        no_cost = (200, {"content-type": "application/json"}, json.dumps({
            "data": [{"b64_json": base64.b64encode(self.output_bytes).decode()}], "usage": {"total_tokens": 1},
        }).encode())
        fake = MockTransport([no_cost])
        with self.assertRaisesRegex(RuntimeError, "no retry"):
            self.execute(True, fake)
        self.assertEqual(len(fake.post_calls), 1)
        terminal = json.loads((self.exec_root / "live/source_a/TERMINAL.json").read_text())
        self.assertEqual(terminal["status"], "SUCCEEDED")
        self.assertFalse(terminal["continue_allowed"])
        self.assertIn("ACTUAL_COST_UNKNOWN_RESERVED_AT_UNIT_CAP", terminal["actual_cost_state"])
        with self.assertRaisesRegex(RuntimeError, "cost/evidence is incomplete"):
            self.execute(True, fake)
        self.assertEqual(len(fake.post_calls), 1)

    def test_review_atlas_exposes_batch_approve_and_mask_views(self):
        atlas = self.private / "atlas"
        atlas.mkdir()
        rows = [{
            "source_id": "fake_1", "class": "Crack", "image_width": "8", "image_height": "6",
            "prompt_id": "CRACK_RESTORATION_V1", "selected_polygon_ids": "fake_1_000",
            "atlas_image_path": "inputs/fake_1.png", "atlas_overlay_path": "previews/fake_1.jpg",
            "mask_path": "masks/fake_1.png", "source_image_sha256": "a" * 64,
            "prompt_file_sha256": "b" * 64,
        }]
        build_review_page(atlas, rows)
        page = (atlas / "index.html").read_text(encoding="utf-8")
        self.assertIn("Approve all pending", page)
        self.assertIn("APPROVE", page)
        self.assertIn("HOLD", page)
        self.assertIn("REJECT", page)
        self.assertIn("g3Rows", page)
        self.assertIn("g3b2b_review_", page)
        self.assertIn("masks/fake_1.png", page)

    def test_corrosion_selector_excludes_base_groups_and_varies_area(self):
        def source(sid: str, group: str) -> dict[str, str]:
            return {"source_id": sid, "user_decision": "SELECT", "repair_priority": "A",
                    "polygon_ids": sid + "_000", "selected_polygon_ids": sid + "_000",
                    "classes_present": "corrosion", "group_id": group,
                    "near_duplicate_group_id": "UNKNOWN", "historical_cloud_exposure": "NONE_RECORDED_IN_G1_G2_AUDIT_MAP"}
        sources = [source(str(i), "g" + str(i)) for i in range(1, 15)]
        instances = {row["source_id"] + "_000": {"candidate_id": row["source_id"] + "_000",
                    "padding_risk": "LOW", "area_fraction_polygon": str(i / 100)}
                    for i, row in enumerate(sources, start=1)}
        base = [{"group_id": "g1", "near_duplicate_group_id": "UNKNOWN"}]
        selected = choose_corrosion(sources, instances, base, 6)
        ids = [item["source"]["source_id"] for item in selected]
        self.assertEqual(len(ids), 6)
        self.assertNotIn("1", ids)
        self.assertEqual(len({item["dedup_key"] for item in selected}), 6)
        areas = [item["area_fraction"] for item in selected]
        self.assertGreater(max(areas) - min(areas), 0.04)


if __name__ == "__main__":
    unittest.main()
