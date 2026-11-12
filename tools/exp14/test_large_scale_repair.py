"""Regression coverage for Exp14 POST identity and raw-to-source overlap."""
import base64
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from tools.exp13_cloud.g3 import align_g3b2_batch as alignment
from tools.exp14 import run_large_scale_repair as runner
from tools.exp14.prepare_large_scale_repair import request_identity


class RequestSafety(unittest.TestCase):
    def setUp(self):
        self.row = {
            "request_id": "abc", "variant_id": "17_V1", "source_id": "17",
            "model": "request-model", "endpoint": "https://example.test/v1/images/edits",
        }
        self.input = {"prompt_wire": b"repair", "image_path": Path("17.png"), "image_bytes": b"image",
                      "image_format": "PNG", "mask_bytes": b"mask"}

    def test_delivery_unknown_consumes_only_one_post(self):
        transport = Mock()
        transport.post.side_effect = TimeoutError("timeout")
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "input_for_row", return_value=self.input):
            args = (Path(temp), self.row, {}, "test-key", transport, "auth", "protocol")
            first = runner.run_one(*args)
            second = runner.run_one(*args)
            self.assertTrue(first["delivery_uncertain"])
            self.assertEqual(second["status"], "LOCKED_NO_POST")
            self.assertEqual(len(list((Path(temp) / "locks").glob("*.json"))), 1)
            transport.post.assert_called_once()

    def test_parallel_dispatchers_share_the_same_permanent_slot(self):
        transport = Mock()
        transport.post.side_effect = TimeoutError("timeout")
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "input_for_row", return_value=self.input):
            args = (Path(temp), self.row, {}, "test-key", transport, "auth", "protocol")
            with ThreadPoolExecutor(max_workers=2) as workers:
                jobs = [workers.submit(runner.run_one, *args) for _ in range(2)]
                statuses = {job.result()["status"] for job in jobs}
            self.assertEqual(statuses, {"FAILED_NO_RETRY", "LOCKED_NO_POST"})
            transport.post.assert_called_once()

    def test_returned_alias_preserves_one_image_and_metadata(self):
        image = io.BytesIO()
        Image.new("RGB", (12, 10)).save(image, format="PNG")
        transport = Mock()
        transport.post.return_value = (200, {"Content-Type": "application/json"}, b"response")
        response = {"image": {"format": "PNG"}, "response_model": "alias", "reported_cost": None,
                    "raw_output_sha256": runner.g3.digest(image.getvalue())}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(runner, "input_for_row", return_value=self.input), \
                patch.object(runner.g3, "parse_output", return_value=(image.getvalue(), response)):
            root = Path(temp)
            result = runner.run_one(root, self.row, {}, "test-key", transport, "auth", "protocol")
            evidence = runner.load_json(root / "live/17_V1/response_manifest.json")
            self.assertEqual(result["status"], "RETURNED_IMAGE")
            self.assertEqual(evidence["model_metadata"], "DIFFERENT")
            self.assertEqual(evidence["response_model"], "alias")
            self.assertEqual(evidence["requested_model"], self.row["model"])
            self.assertEqual((root / "live/17_V1/raw_output.png").read_bytes(), image.getvalue())

    def test_identity_depends_on_wire_semantics(self):
        first = {**self.row, "source_image_sha256": "input", "target_alpha_sha256": "alpha",
                 "prompt_wire_sha256": "prompt", "n": "1", "mask_sha256": "encoder1", "version": "V1"}
        self.assertEqual(request_identity(first), request_identity({**first, "mask_sha256": "encoder2", "version": "V2"}))
        for key in ("source_image_sha256", "target_alpha_sha256", "prompt_wire_sha256", "endpoint", "model", "n"):
            self.assertNotEqual(request_identity(first), request_identity({**first, key: "changed"}))

    def test_unsolicited_second_image_keeps_one_request_and_two_positions(self):
        outputs = []
        for color in ("red", "blue"):
            image = io.BytesIO()
            Image.new("RGB", (12, 10), color).save(image, format="PNG")
            outputs.append({"b64_json": base64.b64encode(image.getvalue()).decode("ascii")})
        payload = json.dumps({"data": outputs, "usage": {"total_tokens": 11}}).encode("utf-8")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "live/17_V1"
            folder.mkdir(parents=True)
            (folder / "response_body.bin").write_bytes(payload)
            runner.g3.write_json_exclusive(folder / "http_response.json", {
                "http_status": 200, "headers": {"content-type": "application/json"},
                "response_body_sha256": runner.g3.digest(payload),
            })
            runner.g3.write_json_exclusive(folder / "TERMINAL.json", {
                "error": "Expected exactly one output image; source slot consumed", "elapsed_seconds": 1,
            })
            runner.recover_responses(root, {"requests": [self.row]})
            runner.recover_responses(root, {"requests": [self.row]})
            evidence = runner.load_json(root / "response_recovery/17_V1/RECOVERED.json")
            self.assertEqual(evidence["api_posts"], 0)
            self.assertEqual(evidence["provider_returned_positions"], 2)
            self.assertEqual([item["position"] for item in evidence["outputs"]], [1, 2])
            self.assertEqual(len({item["raw_sha256"] for item in evidence["outputs"]}), 2)


class AffineOverlap(unittest.TestCase):
    def test_positive_warp_overlap_uses_raw_to_source(self):
        raw_size, source_size = (100, 80), (160, 120)
        matrix = np.array([[1.4, 0., -20.], [0., 1.4, 10.]])
        raw_points = np.array([(x, y) for x in (20, 45, 70, 95) for y in (10, 35, 70)])
        source_points = alignment.project(matrix, raw_points)
        pairs = [{"output": p, "source": q} for p, q in zip(raw_points, source_points)]
        result, _ = alignment.evaluate_matrix(matrix, np.ones((len(pairs), 1)), pairs, raw_size, source_size, "KNOWN")
        expected = (120 * 110) / (160 * 120)
        self.assertAlmostEqual(result["valid_overlap_fraction_estimate"], expected, places=6)
        self.assertTrue(result["criteria"]["valid_overlap_at_least_half"])
        raster = cv2.warpAffine(np.ones(raw_size[::-1], np.uint8), matrix, source_size, flags=cv2.INTER_NEAREST)
        self.assertLess(abs(float(raster.mean()) - expected), 0.015)
        fits = alignment.fit_models(pairs, raw_size, source_size)
        self.assertTrue(all(fit["status"] == "GEOMETRY_CANDIDATE" for fit, _ in fits))


if __name__ == "__main__":
    unittest.main()
