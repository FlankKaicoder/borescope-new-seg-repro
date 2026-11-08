"""Manifest-driven G3 image executor; dry-run by default and at-most-once live."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import http.client
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

from tools.exp13_cloud.g3.prepare_g3b2b_batch import REVIEW_FIELDS
from tools.exp13_cloud.g3.verify_train_export import sha256_file

PROTOCOL_PATH = Path("configs/exp13_cloud/g2/pilot_protocol_v2.json")
BUNDLE_PATH = Path("configs/exp13_cloud/g2/execution_bundle.json")
MAX_RESPONSE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_DIMENSION = 20000
MAX_COST_FIELDS = {"unit_price_cap", "total_spend_cap", "currency", "price_evidence_sha256", "provider_hard_limit_evidence"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_exclusive(path: Path, value: Any) -> None:
    write_exclusive(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path.name}")
        return list(reader.fieldnames), list(reader)


def load_protocol(repo_root: Path) -> tuple[dict[str, Any], str]:
    protocol_path = repo_root / PROTOCOL_PATH
    bundle_path = repo_root / BUNDLE_PATH
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8-sig"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    entry = next((item for item in bundle.get("files", []) if item.get("path") == PROTOCOL_PATH.as_posix()), None)
    require(entry is not None and digest(protocol_bytes) == entry.get("sha256"), "G2 protocol is not byte-identical to frozen bundle")
    request = protocol.get("request", {})
    require(protocol.get("record_type") == "G2_EXECUTION_PROTOCOL_V2_NOT_AUTHORIZATION", "Unexpected frozen protocol type")
    require(request.get("method") == "POST" and request.get("n") == 1, "Frozen G2 request contract changed")
    require(request.get("automatic_retries") == 0 and request.get("fallbacks") == 0 and request.get("redirects") is False, "G2 retry/redirect policy changed")
    require(request.get("multipart_fields") == ["model", "prompt", "n", "image[]", "mask"], "G2 multipart contract changed")
    require(request.get("endpoint") == protocol.get("request", {}).get("endpoint"), "G2 endpoint missing")
    return protocol, digest(protocol_bytes)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    fields, rows = read_csv(path)
    required = {
        "source_id", "split", "class", "decision", "priority", "selected_polygon_ids", "source_image_sha256",
        "source_json_sha256", "image_width", "image_height", "image_path", "batch_image_path", "mask_path",
        "mask_sha256", "prompt_id", "prompt_file", "prompt_file_sha256", "prompt_wire_sha256", "status",
        "manually_approved", "authorized_for_paid_post", "request_count_proposed", "roi_policy", "n",
    }
    require(required <= set(fields), "Expanded G3 manifest is missing executor fields")
    ids = [row["source_id"] for row in rows]
    require(len(ids) == len(set(ids)), "Manifest source IDs are duplicated")
    require(all(row.get("split") == "train" and row.get("status") == "PROPOSED" for row in rows), "Manifest is not a TRAIN proposal set")
    return fields, rows, sha256_file(path)


def safe_file(root: Path, relative: str) -> Path:
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, "Unsafe manifest path")
    path = (root / value).resolve()
    require(root.resolve() in path.parents and path.is_file(), f"Input file missing or outside batch root: {relative}")
    return path


def validate_review(review_path: Path, rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    fields, reviews = read_csv(review_path)
    require(set(REVIEW_FIELDS) <= set(fields), "Review CSV schema mismatch")
    by_id = {row["source_id"]: row for row in reviews}
    require(len(by_id) == len(reviews) == len(rows), "Review CSV source count or uniqueness mismatch")
    if set(by_id) != {row["source_id"] for row in rows}:
        raise ValueError("Review CSV source exact-set differs from manifest")
    allowed = {"PENDING", "APPROVE", "HOLD", "REJECT"}
    for row in rows:
        review = by_id[row["source_id"]]
        require(review["source_image_sha256"] == row["source_image_sha256"], "Review source image hash mismatch")
        require(review["prompt_id"] == row["prompt_id"] and review["prompt_file_sha256"] == row["prompt_file_sha256"], "Review Prompt identity mismatch")
        require(review["review_status"] in allowed, "Invalid human review status")
    return reviews, sha256_file(review_path)


def validate_input(row: dict[str, str], batch_root: Path, repo_root: Path) -> dict[str, Any]:
    require(row["decision"] == "SELECT" and row["split"] == "train", "Only TRAIN/SELECT sources are executable")
    require(row["status"] == "PROPOSED" and row["manually_approved"] == "NO" and row["authorized_for_paid_post"] == "NO", "Manifest approval state is inconsistent; authorization is separate")
    require(row["request_count_proposed"] == "1" and row["n"] == "1", "Exactly one output per source is required")
    require(row["roi_policy"] == "FULL_NATIVE_FRAME_NO_CROP_RESIZE_PADDING_OR_COORDINATE_TRANSFORM", "Only FULL_NATIVE_FRAME inputs are supported")
    image_path = safe_file(batch_root, row["batch_image_path"])
    mask_path = safe_file(batch_root, row["mask_path"])
    prompt_relative = Path(row["prompt_file"])
    require(not prompt_relative.is_absolute() and ".." not in prompt_relative.parts, "Unsafe Prompt path")
    prompt_path = (repo_root / prompt_relative).resolve()
    require(repo_root.resolve() in prompt_path.parents and prompt_path.is_file(), "Prompt file missing or outside repository")
    require(sha256_file(image_path) == row["source_image_sha256"], "Source image SHA256 mismatch")
    require(sha256_file(mask_path) == row["mask_sha256"], "RGBA Mask SHA256 mismatch")
    prompt_bytes = prompt_path.read_bytes()
    require(digest(prompt_bytes) == row["prompt_file_sha256"], "Prompt file SHA256 mismatch")
    require(prompt_bytes.endswith(b"\n") and not prompt_bytes.endswith(b"\n\n") and not prompt_bytes.startswith(b"\xef\xbb\xbf"), "Prompt bytes must have one terminal LF and no BOM")
    prompt_wire = prompt_bytes[:-1]
    require(digest(prompt_wire) == row["prompt_wire_sha256"], "Prompt wire SHA256 mismatch")
    with Image.open(image_path) as image:
        image.load()
        image_size = image.size
        image_format = image.format
    width, height = int(row["image_width"]), int(row["image_height"])
    require(image_size == (width, height) and max(image_size) <= MAX_IMAGE_DIMENSION, "Input image resolution differs from Manifest")
    if image_format not in {"PNG", "JPEG"}:
        raise ValueError("G2 protocol only has PNG/JPEG input MIME handling")
    with Image.open(mask_path) as mask_image:
        mask_image.load()
        require(mask_image.mode == "RGBA" and mask_image.size == image_size, "Mask must be native-size RGBA PNG")
        alpha = mask_image.getchannel("A")
        histogram = alpha.histogram()
        require(histogram[0] > 0 and histogram[255] > 0 and sum(histogram[1:255]) == 0, "Mask alpha must target transparent pixels and preserve opaque context")
    return {
        "image_path": image_path, "mask_path": mask_path, "prompt_path": prompt_path,
        "image_bytes": image_path.read_bytes(), "mask_bytes": mask_path.read_bytes(),
        "prompt_wire": prompt_wire, "image_format": image_format, "image_size": image_size,
        "mask_sha256": row["mask_sha256"], "prompt_file_sha256": digest(prompt_bytes),
        "prompt_wire_sha256": digest(prompt_wire),
    }


def encode_multipart(source_id: str, model: str, prompt_wire: bytes, image_name: str, image_bytes: bytes, image_format: str, mask_bytes: bytes) -> tuple[bytes, str]:
    boundary = f"borescope_g3_{source_id}_{digest(image_bytes + mask_bytes + prompt_wire)[:16]}"
    parts = [
        ("model", model.encode("utf-8"), None, None), ("prompt", prompt_wire, None, None), ("n", b"1", None, None),
        ("image[]", image_bytes, image_name, "image/png" if image_format == "PNG" else "image/jpeg"),
        ("mask", mask_bytes, "api_rgba_mask.png", "image/png"),
    ]
    content: list[bytes] = []
    for name, value, filename, mime in parts:
        header = f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\""
        if filename:
            header += f"; filename=\"{filename}\"\r\nContent-Type: {mime}"
        content.extend([(header + "\r\n\r\n").encode("utf-8"), value, b"\r\n"])
    body = b"".join(content) + f"--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def safe_header_summary(headers: dict[str, str]) -> dict[str, str]:
    allowed = {"content-type", "content-length", "x-request-id", "request-id"}
    return {key.lower(): value for key, value in headers.items() if key.lower() in allowed}


class HttpTransport:
    def post(self, endpoint: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, str], bytes]:
        parsed = urlsplit(endpoint)
        require(parsed.scheme == "https" and parsed.hostname and parsed.port in (None, 443), "Only HTTPS port 443 endpoint is allowed")
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=600)
        try:
            connection.request("POST", parsed.path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, {k: v for k, v in response.getheaders()}, response.read(MAX_RESPONSE_BYTES + 1)
        finally:
            connection.close()

    def get(self, url: str) -> tuple[int, dict[str, str], bytes]:
        parsed = urlsplit(url)
        require(parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password and parsed.port in (None, 443), "Output URL is not allowed")
        require(parsed.hostname.lower() not in {"localhost"} and not parsed.hostname.lower().endswith(".local"), "Local output host rejected")
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=120)
        try:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            connection.request("GET", path, headers={"Accept": "image/*"})
            response = connection.getresponse()
            return response.status, {k: v for k, v in response.getheaders()}, response.read(MAX_RESPONSE_BYTES + 1)
        finally:
            connection.close()


def parse_output(status: int, headers: dict[str, str], response_body: bytes, transport: Any) -> tuple[bytes, dict[str, Any]]:
    if len(response_body) > MAX_RESPONSE_BYTES:
        raise ValueError("Response exceeds maximum size; source slot consumed")
    if not 200 <= status < 300:
        raise ValueError(f"HTTP {status}; source slot consumed and no retry")
    content_type = next((value for key, value in headers.items() if key.lower() == "content-type"), "")
    payload: dict[str, Any]
    if content_type.lower().startswith("image/"):
        raw = response_body
        payload = {}
        response_format = "direct_image"
    else:
        payload = json.loads(response_body.decode("utf-8"))
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != 1:
            raise ValueError("Expected exactly one output image; source slot consumed")
        item = data[0]
        if item.get("b64_json"):
            raw = base64.b64decode(item["b64_json"], validate=True)
            response_format = "b64_json"
        elif item.get("url"):
            get_status, get_headers, raw = transport.get(item["url"])
            if get_status != 200 or any(key.lower() == "location" for key in get_headers):
                raise ValueError("Image URL GET failed or attempted redirect; source slot consumed")
            response_format = "url_download"
        else:
            raise ValueError("Response has no image bytes or URL; source slot consumed")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Decoded image exceeds maximum size; source slot consumed")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        info = {"format": image.format, "mode": image.mode, "size_wh": list(image.size)}
    if info["format"] not in {"PNG", "JPEG", "WEBP"}:
        raise ValueError("Unsupported output image format; source slot consumed")
    payload_safe = {"model": payload.get("model"), "usage": payload.get("usage"), "cost": payload.get("cost")}
    usage = payload_safe["usage"] if isinstance(payload_safe["usage"], dict) else None
    cost = payload_safe["cost"]
    if cost is None and usage is not None:
        cost = usage.get("cost")
    return raw, {
        "response_format": response_format, "response_model": payload_safe["model"], "usage": usage,
        "reported_cost": cost, "cost_status": "PROVIDER_REPORTED" if cost is not None else "NOT_RETURNED",
        "image": info, "raw_output_sha256": digest(raw), "response_body_sha256": digest(response_body),
        "response_body_bytes": len(response_body),
    }


def validate_authorization(auth_path: Path, auth: dict[str, Any], manifest_sha: str, review_sha: str,
                           prompt_manifest_sha: str, protocol_sha: str, protocol: dict[str, Any],
                           approved: list[str], rows_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    request = protocol["request"]
    require(auth.get("record_type") == "G3B2B_LIVE_AUTHORIZATION" and auth.get("status") == "G3B2B_BATCH_LIVE_APPROVED", "Live authorization record missing or wrong status")
    user_direct = auth.get("authorization_basis") == "DIRECT_USER_BATCH_AUTHORIZATION_NO_PRICE_CAP"
    require(auth.get("user_approved") is True, "User batch authorization is required")
    if not user_direct:
        require(auth.get("independent_gate_pass") is True, "Independent Gate approval is required")
    for key, expected in (("batch_manifest_sha256", manifest_sha), ("review_csv_sha256", review_sha),
                          ("prompt_manifest_sha256", prompt_manifest_sha), ("protocol_sha256", protocol_sha)):
        require(auth.get(key) == expected, f"Authorization {key} mismatch")
    approved_ids = auth.get("approved_source_ids")
    require(isinstance(approved_ids, list) and len(approved_ids) == len(set(approved_ids)) and set(approved_ids) == set(approved), "Authorization source exact-set differs from APPROVE review rows")
    require(approved_ids and all(source_id in rows_by_id for source_id in approved_ids), "No approved sources or source absent from Manifest")
    require(auth.get("request_count_cap") == len(approved_ids) and len(approved_ids) <= len(rows_by_id), "Request count cap must equal the authorized exact source set")
    require(auth.get("provider") == request["base_url"] and auth.get("model") == request["model"] and auth.get("endpoint") == request["endpoint"], "Authorization provider/model/endpoint mismatch")
    require(auth.get("n") == 1, "Only one output image per source is supported")
    if user_direct:
        require(auth.get("cost_status") == "COST_UNKNOWN_USER_AUTHORIZED_WITHOUT_SPEND_CAP", "Direct authorization cost state must remain explicitly unknown")
        require(auth.get("user_authorization_text") == "直接调用即可，不需要计算费用区间；授权操作", "Direct user authorization text mismatch")
        require(auth.get("max_posts") == len(approved_ids), "Direct authorization POST maximum must equal its exact source set")
    else:
        missing = MAX_COST_FIELDS - set(auth)
        require(not missing, "Authorization missing price evidence/cost caps")
        require(auth.get("currency") and re.fullmatch(r"[A-Z]{3}", str(auth["currency"])), "Currency must be an ISO 4217 code")
        require(re.fullmatch(r"[0-9a-f]{64}", str(auth.get("price_evidence_sha256", ""))) is not None, "Price evidence SHA256 missing")
        require(isinstance(auth.get("provider_hard_limit_evidence"), str) and auth["provider_hard_limit_evidence"].strip(), "Provider-enforced cap evidence is required")
        unit_cap, total_cap = float(auth["unit_price_cap"]), float(auth["total_spend_cap"])
        require(unit_cap > 0 and total_cap >= unit_cap * len(approved_ids), "Cost caps are invalid or below the authorized request envelope")
        supported = {tuple(value) for value in auth.get("supported_input_resolutions", []) if isinstance(value, list) and len(value) == 2}
        require(all((int(rows_by_id[source_id]["image_width"]), int(rows_by_id[source_id]["image_height"])) in supported for source_id in approved_ids), "Provider resolution evidence does not cover every approved source")
    reference = auth.get("approval_reference")
    require(isinstance(reference, str) and reference.strip(), "Approval reference is required")
    if user_direct:
        return {"sha256": sha256_file(auth_path), "unit_price_cap": None, "total_spend_cap": None,
                "currency": None, "cost_unknown_authorized": True, "approved_ids": approved_ids}
    return {"sha256": sha256_file(auth_path), "unit_price_cap": unit_cap, "total_spend_cap": total_cap,
            "currency": auth["currency"], "cost_unknown_authorized": False, "approved_ids": approved_ids}


def terminal_state(lock_path: Path, source_dir: Path) -> str:
    if not lock_path.exists():
        if source_dir.exists():
            raise RuntimeError("Existing source output without request lock; preserve and inspect, never overwrite")
        return "NOT_STARTED"
    terminal = source_dir / "TERMINAL.json"
    if not terminal.is_file():
        raise RuntimeError("Permanent request lock has no terminal record; delivery is uncertain, stop without retry")
    record = json.loads(terminal.read_text(encoding="utf-8"))
    if record.get("status") != "SUCCEEDED":
        raise RuntimeError("Prior request ended non-successfully; source and batch remain stopped")
    if record.get("continue_allowed") is not True:
        raise RuntimeError("Prior request cost/evidence is incomplete; batch remains stopped")
    return "SUCCEEDED"


def reserve_lock(lock_path: Path, payload: dict[str, Any]) -> None:
    write_json_exclusive(lock_path, payload)


def run_batch(manifest_path: Path, review_path: Path, batch_root: Path, execution_root: Path,
              repo_root: Path, private_root: Path, live: bool = False, auth_path: Path | None = None,
              api_key: Any = None, transport: Any | None = None) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(repo_root)
    fields, rows, manifest_sha = read_manifest(manifest_path)
    reviews, review_sha = validate_review(review_path, rows)
    review_by_id = {row["source_id"]: row for row in reviews}
    prompt_manifest_path = batch_root / "prompt_manifest.json"
    prompt_manifest_sha = sha256_file(prompt_manifest_path)
    prompt_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8-sig"))
    prompt_index = prompt_manifest.get("prompts", {})
    rows_by_id = {row["source_id"]: row for row in rows}
    for row in rows:
        prompt = prompt_index.get(row["class"])
        require(isinstance(prompt, dict), f"Prompt class missing from batch Prompt manifest: {row['class']}")
        require(prompt.get("prompt_id") == row["prompt_id"] and
                prompt.get("path") == row["prompt_file"] and
                prompt.get("file_sha256") == row["prompt_file_sha256"] and
                prompt.get("wire_sha256") == row["prompt_wire_sha256"],
                f"Source Prompt differs from batch Prompt manifest: {row['source_id']}")
    inputs = {row["source_id"]: validate_input(row, batch_root, repo_root) for row in rows}
    approved = [row["source_id"] for row in rows if review_by_id[row["source_id"]]["review_status"] == "APPROVE"]
    auth = None
    if live:
        if auth_path is None or not auth_path.is_file():
            raise ValueError("Live mode requires a separate private authorization record")
        auth_value = json.loads(auth_path.read_text(encoding="utf-8-sig"))
        if auth_value.get("authorization_basis") == "DIRECT_USER_BATCH_AUTHORIZATION_NO_PRICE_CAP":
            approved = auth_value.get("approved_source_ids", [])
            require(all(review_by_id.get(source_id, {}).get("review_status") in {"PENDING", "APPROVE"} for source_id in approved),
                    "Direct batch authorization includes a held or rejected source")
        auth = validate_authorization(auth_path, auth_value, manifest_sha, review_sha, prompt_manifest_sha,
                                      protocol_sha, protocol, approved, rows_by_id)
        transport = transport or HttpTransport()
        target_ids = auth["approved_ids"]
    else:
        target_ids = [row["source_id"] for row in rows]

    execution_root = execution_root.resolve()
    private_root = private_root.resolve()
    require(private_root in execution_root.parents, "Execution records must stay under the private asset root")
    require(private_root in batch_root.resolve().parents, "Batch inputs must stay under the private asset root")
    if batch_root.resolve() in execution_root.parents or repo_root.resolve() in execution_root.parents:
        raise ValueError("Execution output must be separate from repository and batch input roots")
    run_root = execution_root / ("live" if live else "dry_run")
    run_root.mkdir(parents=True, exist_ok=True)
    credential = api_key() if live and callable(api_key) else api_key
    if live and (not credential or not str(credential).strip()):
        raise ValueError("Live mode credential is missing")
    locks = execution_root / "locks"
    results = []
    cumulative_spend = 0.0
    unknown_cost_ids: list[str] = []
    sent_count = 0
    by_id = {row["source_id"]: row for row in rows}
    for sid in target_ids:
        row = by_id[sid]
        inp = inputs[sid]
        review = review_by_id[sid]
        if live and review["review_status"] not in ({"PENDING", "APPROVE"} if auth["cost_unknown_authorized"] else {"APPROVE"}):
            raise RuntimeError("Source review changed after authorization; stop")
        source_dir = run_root / sid
        lock_path = locks / f"{sid}.json"
        if live:
            state = terminal_state(lock_path, source_dir)
            if state == "SUCCEEDED":
                results.append({"source_id": sid, "status": "ALREADY_SUCCEEDED_NO_POST"})
                continue
        body, content_type = encode_multipart(sid, protocol["request"]["model"], inp["prompt_wire"],
                                              inp["image_path"].name, inp["image_bytes"], inp["image_format"], inp["mask_bytes"])
        request_record = {
            "source_id": sid, "method": "POST", "endpoint": protocol["request"]["base_url"] + protocol["request"]["endpoint"],
            "model": protocol["request"]["model"], "n": 1, "multipart_fields": protocol["request"]["multipart_fields"],
            "content_type": content_type, "body_sha256": digest(body), "body_bytes": len(body),
            "input_sha256": row["source_image_sha256"], "input_size_wh": list(inp["image_size"]),
            "input_aspect_ratio": round(inp["image_size"][0] / inp["image_size"][1], 8),
            "input_format": inp["image_format"], "mask_sha256": inp["mask_sha256"],
            "prompt_id": row["prompt_id"], "prompt_file_sha256": inp["prompt_file_sha256"],
            "prompt_wire_sha256": inp["prompt_wire_sha256"], "protocol_sha256": protocol_sha,
            "request_count_for_source": 1, "automatic_retries": 0, "redirects_followed": False,
            "review_status": review["review_status"], "authorization_sha256": auth["sha256"] if auth else None,
        }
        if not live:
            dry_dir = run_root / sid
            dry_dir.mkdir(exist_ok=False)
            write_json_exclusive(dry_dir / "request_preview.json", {
                "status": "DRY_RUN_READY_NOT_AUTHORIZED", "api_post_count": 0,
                "authorization_created": False, "request": request_record,
            })
            results.append({"source_id": sid, "status": "DRY_RUN_READY_NOT_AUTHORIZED", "post_count": 0})
            continue

        source_dir.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(source_dir / "request_manifest.json", request_record)
        lock_payload = {
            "status": "PERMANENT_SLOT_CONSUMED_BEFORE_POST", "source_id": sid,
            "created_utc": datetime.now(timezone.utc).isoformat(), "request_sha256": digest(json.dumps(request_record, sort_keys=True).encode()),
            "authorization_sha256": auth["sha256"], "rule": "Never delete or reuse; uncertain delivery consumes the source slot.",
        }
        reserve_lock(lock_path, lock_payload)
        posted = False
        start = time.monotonic()
        response_status: int | None = None
        response_headers: dict[str, str] = {}
        response_body = b""
        try:
            endpoint = request_record["endpoint"]
            posted = True
            response_status, response_headers, response_body = transport.post(endpoint, {
                "Authorization": "Bearer " + str(credential), "Content-Type": content_type,
                "Content-Length": str(len(body)), "Accept": "application/json,image/*",
            }, body)
            safe_headers = safe_header_summary(response_headers)
            raw_output, response_record = parse_output(response_status, response_headers, response_body, transport)
            if response_record.get("response_model") not in (None, request_record["model"]):
                raise RuntimeError("Response model identity differs from authorized model")
            write_json_exclusive(source_dir / "http_response.json", {
                "http_status": response_status, "headers": safe_headers,
                "response_body_sha256": digest(response_body), "response_body_bytes": len(response_body),
                "post_invocations": 1, "redirect_followed": False,
            })
            ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}[response_record["image"]["format"]]
            write_exclusive(source_dir / "raw_output.bin", raw_output)
            write_exclusive(source_dir / f"raw_output.{ext}", raw_output)
            response_record.update({
                "http_status": response_status, "headers": safe_headers,
                "output_size_wh": response_record["image"]["size_wh"],
                "output_aspect_ratio": round(response_record["image"]["size_wh"][0] / response_record["image"]["size_wh"][1], 8),
                "elapsed_seconds": round(time.monotonic() - start, 6), "post_invocations": 1,
                "train_pair_eligible": False,
            })
            write_json_exclusive(source_dir / "response_manifest.json", response_record)
            cost = response_record.get("reported_cost")
            if cost is None:
                if auth["cost_unknown_authorized"]:
                    cost_state = "COST_UNKNOWN_NO_PROVIDER_AMOUNT_RETURNED"
                    unknown_cost_ids.append(sid)
                else:
                    cumulative_spend += auth["unit_price_cap"]
                    cost_state = "ACTUAL_COST_UNKNOWN_RESERVED_AT_UNIT_CAP"
            else:
                cost_number = float(cost)
                if cost_number < 0 or (auth["unit_price_cap"] is not None and cost_number > auth["unit_price_cap"]):
                    raise RuntimeError("Reported per-request cost exceeds authorization unit cap")
                cumulative_spend += cost_number
                cost_state = "PROVIDER_REPORTED"
            if auth["total_spend_cap"] is not None and cumulative_spend > auth["total_spend_cap"]:
                raise RuntimeError("Cumulative cost cap exceeded; stop batch")
            terminal = {"status": "SUCCEEDED", "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "post_invocations": 1, "actual_cost_state": cost_state,
                        "cumulative_reported_or_reserved_cost": round(cumulative_spend, 8) if not auth["cost_unknown_authorized"] else None,
                        "currency": auth["currency"], "continue_allowed": cost is not None or auth["cost_unknown_authorized"]}
            write_json_exclusive(source_dir / "TERMINAL.json", terminal)
            sent_count += 1
            results.append({"source_id": sid, "status": "SUCCEEDED", "post_count": 1,
                            "cost_state": cost_state,
                            "cumulative_cost": round(cumulative_spend, 8) if not auth["cost_unknown_authorized"] else None})
            if cost is None and not auth["cost_unknown_authorized"]:
                raise RuntimeError("Provider omitted per-call cost; stop after this call to preserve total-spend accounting")
        except Exception as error:
            if response_status is not None and not (source_dir / "http_response.json").exists():
                write_json_exclusive(source_dir / "http_response.json", {
                    "http_status": response_status, "headers": safe_header_summary(response_headers),
                    "response_body_sha256": digest(response_body), "response_body_bytes": len(response_body),
                    "post_invocations": 1, "redirect_followed": False,
                    "usage": None, "reported_cost": None, "cost_status": "NOT_RETURNED_OR_UNPARSEABLE",
                })
            if not (source_dir / "TERMINAL.json").exists():
                write_json_exclusive(source_dir / "TERMINAL.json", {
                    "status": "FAILED_NO_RETRY", "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "post_invoked": posted, "delivery_or_billing": "UNKNOWN" if posted else "NOT_SENT",
                    "error_type": type(error).__name__, "error_message": str(error)[:300],
                    "automatic_retries": 0,
                })
            raise RuntimeError(f"G3 source stopped after one attempt ({sid}); no retry; inspect private evidence") from None

    summary = {
        "status": "DRY_RUN_READY_NOT_AUTHORIZED" if not live else "LIVE_BATCH_TERMINAL",
        "manifest_sha256": manifest_sha, "review_csv_sha256": review_sha,
        "protocol_sha256": protocol_sha, "prompt_manifest_sha256": prompt_manifest_sha,
        "selected_source_count": len(target_ids), "api_post_count": sent_count,
        "authorization_present": bool(auth),
        "cost_status": ("NOT_APPLICABLE_NO_POST" if not live else
                        "COST_UNKNOWN" if unknown_cost_ids else
                        "PROVIDER_REPORTED_NO_SPEND_CAP" if auth and auth["cost_unknown_authorized"] else
                        "COST_REPORTED_OR_CAPPED"),
        "provider_reported_cost_sum": round(cumulative_spend, 8),
        "provider_reported_cost_sum_is_complete": not unknown_cost_ids,
        "cost_unknown_source_ids": unknown_cost_ids,
        "results": results,
    }
    write_json_exclusive(run_root / f"run_summary_{time.time_ns()}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="Requires a separate private exact-set paid authorization")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_batch(args.manifest.resolve(), args.review_csv.resolve(), args.batch_root.resolve(),
                               args.execution_root.resolve(), args.repo_root.resolve(), args.private_root.resolve(), args.live,
                               args.authorization.resolve() if args.authorization else None,
                               (lambda: os.environ.get("G3_IMAGE_API_KEY")) if args.live else None),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOP_G3_BATCH: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        raise SystemExit(1)
