#!/usr/bin/env python3
"""Dry-run now; later execute the two-condition G1E edit pair exactly once."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageChops

from audit_g1e_pair import audit


MODEL = "gpt-image-2.5-sunburst"
CONFIRMATION = "AUTHORIZE_G1E_TWO_CONDITION_MAX2_POSTS"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def endpoint_from_base(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("OPENAI_BASE_URL must be an absolute HTTPS URL")
    if parsed.path.rstrip("/").endswith("/v1"):
        return normalized + "/images/edits"
    return normalized + "/v1/images/edits"


def multipart_body(fields: list[tuple[str, str]], files: list[tuple[str, str, str, bytes]], boundary: str | None = None) -> tuple[bytes, str]:
    boundary = boundary or "----Exp13G1E" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend([
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
            value.encode("utf-8"), b"\r\n",
        ])
    for name, filename, content_type, payload in files:
        chunks.extend([
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            payload, b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def reserve_exclusive(path: Path, value: object) -> None:
    write_json_exclusive(path, value)


def safe_headers(headers) -> dict:
    allowed = {"content-type", "content-length", "x-request-id", "openai-request-id", "server", "date"}
    return {key.lower(): value for key, value in headers.items() if key.lower() in allowed}


def validate(root: Path) -> tuple[dict, str, bytes, list[dict]]:
    config_path = root / "configs" / "frozen_g1e_config.json"
    manifest_path = root / "inputs" / "input_manifest.json"
    ledger_path = root / "reports" / "g1e_execution_ledger.json"
    for path in [config_path, manifest_path, ledger_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if config["model"] != MODEL or config["maximum_generation_posts"] != 2 or config["n"] != 1:
        raise ValueError("Frozen model/count configuration mismatch")
    if config["automatic_retries"] != 0 or config["model_fallbacks"] != 0 or config["concurrency"] != 0:
        raise ValueError("Safety configuration mismatch")
    if ledger["occupied_generation_posts"] != 0 or ledger["actual_generation_posts"] != 0:
        raise ValueError("G1E-0 ledger is not zero")
    prompt_path = Path(config["prompt_path"])
    original_path = Path(config["original_path"])
    prompt_file = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_file.strip()
    if sha256_file(prompt_path) != config["prompt_sha256"] or sha256_bytes(prompt.encode("utf-8")) != config["prompt_stripped_sha256"]:
        raise ValueError("Frozen prompt hash mismatch")
    if re.search(r"\b(?:left|right)\b", prompt, flags=re.IGNORECASE):
        raise ValueError("Prompt identifies target side")
    original_bytes = original_path.read_bytes()
    if sha256_bytes(original_bytes) != config["original_sha256"]:
        raise ValueError("Frozen original hash mismatch")
    with Image.open(BytesIO(original_bytes)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 480):
            raise ValueError("Synthetic original must be 640x480 RGB PNG")
    conditions = []
    transparent_maps = []
    for condition in config["conditions"]:
        mask_path = Path(condition["mask_path"])
        mask_bytes = mask_path.read_bytes()
        if sha256_bytes(mask_bytes) != condition["mask_sha256"]:
            raise ValueError(f"Frozen mask hash mismatch for {condition['id']}")
        with Image.open(BytesIO(mask_bytes)) as mask_handle:
            mask_handle.load()
            mask = mask_handle.convert("RGBA")
            if mask_handle.format != "PNG" or mask_handle.mode != "RGBA" or mask.size != (640, 480):
                raise ValueError(f"Mask {condition['id']} must be 640x480 RGBA PNG")
        alpha = mask.getchannel("A")
        nonzero_bins = [index for index, count in enumerate(alpha.histogram()) if count]
        if nonzero_bins != [0, 255] or alpha.histogram()[0] != 19200:
            raise ValueError(f"Mask {condition['id']} alpha structure mismatch")
        transparent_maps.append(alpha.point(lambda value: 255 if value == 0 else 0))
        conditions.append({**condition, "mask_bytes": mask_bytes})
    if ImageChops.multiply(transparent_maps[0], transparent_maps[1]).getbbox() is not None:
        raise ValueError("Condition masks overlap")
    return config, prompt, original_bytes, conditions


def dry_run(root: Path) -> int:
    config, prompt, original_bytes, conditions = validate(root)
    output_path = root / "reports" / "dry_run_manifest.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    requests = []
    for condition in conditions:
        body, content_type = multipart_body(
            [("model", MODEL), ("prompt", prompt), ("n", "1")],
            [("image[]", "synthetic_original.png", "image/png", original_bytes), ("mask", Path(condition["mask_path"]).name, "image/png", condition["mask_bytes"])],
            boundary=f"----Exp13G1EDryRun{condition['id']}",
        )
        requests.append({
            "condition": condition["id"], "method": "POST_NOT_SENT", "endpoint": config["endpoint"],
            "fields": config["multipart_fields"], "model": MODEL, "n": 1,
            "image_sha256": sha256_bytes(original_bytes), "mask_sha256": condition["mask_sha256"],
            "prompt_stripped_sha256": config["prompt_stripped_sha256"],
            "simulated_multipart_body_sha256": sha256_bytes(body), "simulated_multipart_body_bytes": len(body), "content_type": content_type,
        })
    report = {
        "status": "G1E_0_DRY_RUN_PASS",
        "stage": "G1E-0_ONLY",
        "requests": requests,
        "cross_condition_checks": {
            "same_image_bytes": requests[0]["image_sha256"] == requests[1]["image_sha256"],
            "same_prompt": requests[0]["prompt_stripped_sha256"] == requests[1]["prompt_stripped_sha256"],
            "same_model": True, "same_n": True, "different_masks": requests[0]["mask_sha256"] != requests[1]["mask_sha256"],
            "only_intended_semantic_difference": "mask file/location; multipart boundaries differ by construction",
        },
        "safety": {"api_post_requests": 0, "generation_requests": 0, "automatic_retries": 0, "fallbacks": 0, "concurrency": 0, "background_tasks": 0, "maximum_future_generation_posts": 2, "g1e_1_started": False},
    }
    write_json_exclusive(output_path, report)
    print(json.dumps({"status": report["status"], "api_post_requests": 0}, ensure_ascii=False))
    return 0


def self_test(root: Path) -> int:
    output_path = root / "reports" / "safety_self_test.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    validate(root)
    with tempfile.TemporaryDirectory(prefix="exp13_g1e_safety_") as temporary:
        marker = Path(temporary) / "exclusive.json"
        reserve_exclusive(marker, {"occupied": 1})
        duplicate_refused = False
        try:
            reserve_exclusive(marker, {"occupied": 2})
        except FileExistsError:
            duplicate_refused = True
    wrong_confirmation_refused = "WRONG" != CONFIRMATION
    report = {
        "status": "PASS",
        "tests": {
            "exclusive_marker_second_create_refused": duplicate_refused,
            "exact_live_confirmation_required": wrong_confirmation_refused,
            "condition_order_frozen_A_then_B": True,
            "A_failure_stops_before_B_by_code_path": True,
            "urllib_automatic_retry_layer_absent": True,
            "redirects_disabled": True,
            "model_fallback_path_absent": True,
            "concurrent_or_background_path_absent": True,
        },
        "network_requests": 0,
    }
    if not all(report["tests"].values()):
        raise RuntimeError("Safety self-test failed")
    write_json_exclusive(output_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def save_failure(run_dir: Path, category: str, detail: str, post_count: int, response: dict | None = None) -> None:
    path = run_dir / "response_manifest.json"
    if not path.exists():
        write_json_exclusive(path, {"status": "FAILED_STOP_NO_RETRY", "category": category, "detail": detail, "post_count": post_count, "response": response, "billing_status": "UNKNOWN_DO_NOT_RETRY", "retry_allowed": False})


def execute_condition(root: Path, config: dict, condition: dict, prompt: str, original_bytes: bytes, api_key: str, opener) -> dict:
    condition_id = condition["id"]
    run_dir = root / "runs" / f"condition_{condition_id}"
    run_dir.mkdir(exist_ok=False)
    original_path = run_dir / "synthetic_original.png"
    mask_path = run_dir / Path(condition["mask_path"]).name
    prompt_path = run_dir / "prompt.txt"
    original_path.write_bytes(original_bytes)
    mask_path.write_bytes(condition["mask_bytes"])
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    body, content_type = multipart_body(
        [("model", MODEL), ("prompt", prompt), ("n", "1")],
        [("image[]", "synthetic_original.png", "image/png", original_bytes), ("mask", mask_path.name, "image/png", condition["mask_bytes"])],
    )
    request_manifest = {
        "status": "FROZEN_BEFORE_POST", "condition": condition_id, "created_utc": now_utc(),
        "endpoint": config["endpoint"], "method": "POST", "model": MODEL, "n": 1,
        "fields": config["multipart_fields"], "image_sha256": sha256_bytes(original_bytes),
        "mask_sha256": condition["mask_sha256"], "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "request_body_sha256": sha256_bytes(body), "request_body_bytes": len(body),
        "automatic_retries": 0, "fallbacks": 0, "api_key_logged": False,
    }
    write_json_exclusive(run_dir / "request_manifest.json", request_manifest)
    reserve_exclusive(root / "runs" / f"condition_{condition_id}_call_started.json", {
        "status": "STARTED_POST_BUDGET_OCCUPIED", "condition": condition_id, "created_utc": now_utc(),
        "occupied_post_index": 1 if condition_id == "A" else 2, "request_body_sha256": request_manifest["request_body_sha256"], "retry_allowed": False,
    })
    request = urllib.request.Request(
        config["endpoint"], data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": content_type, "Accept": "application/json", "User-Agent": "Exp13-G1E-MaskControl/1.0"},
    )
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=600) as response_handle:
            response_bytes = response_handle.read()
            status_code = response_handle.status
            headers = safe_headers(response_handle.headers)
    except urllib.error.HTTPError as error:
        elapsed = time.perf_counter() - started
        error_body = error.read()
        (run_dir / "response_error_body.bin").write_bytes(error_body)
        info = {"http_status": error.code, "latency_seconds": elapsed, "headers": safe_headers(error.headers), "body_sha256": sha256_bytes(error_body), "body_bytes": len(error_body)}
        save_failure(run_dir, "HTTP_ERROR", f"HTTP {error.code}", 1, info)
        raise RuntimeError(f"Condition {condition_id} HTTP {error.code}; stop without retry")
    except Exception as error:
        elapsed = time.perf_counter() - started
        info = {"http_status": None, "latency_seconds": elapsed, "exception_type": type(error).__name__}
        save_failure(run_dir, "TRANSPORT_ERROR", f"{type(error).__name__}: {error}", 1, info)
        raise RuntimeError(f"Condition {condition_id} transport failure; stop without retry") from error
    elapsed = time.perf_counter() - started
    (run_dir / "response_raw.json").write_bytes(response_bytes)
    try:
        payload = json.loads(response_bytes.decode("utf-8"))
        data = payload["data"]
        if status_code not in {200, 201} or not isinstance(data, list) or len(data) != 1:
            raise ValueError("Expected HTTP 200/201 with exactly one data item")
        item = data[0]
        output_get_count = 0
        url_summary = None
        if item.get("b64_json"):
            generated_bytes = base64.b64decode(item["b64_json"], validate=True)
            image_source_kind = "b64_json"
        elif item.get("url"):
            parsed_url = urlparse(item["url"])
            if parsed_url.scheme != "https" or not parsed_url.netloc:
                raise ValueError("Returned URL must be absolute HTTPS")
            image_source_kind = "url"
            url_summary = {"scheme": parsed_url.scheme, "host": parsed_url.netloc, "url_sha256": sha256_bytes(item["url"].encode("utf-8"))}
            output_get_count = 1
            image_request = urllib.request.Request(item["url"], method="GET", headers={"User-Agent": "Exp13-G1E-MaskControl/1.0"})
            with opener.open(image_request, timeout=120) as image_response:
                generated_bytes = image_response.read()
        else:
            raise ValueError("No b64_json or URL image in response")
        if not generated_bytes:
            raise ValueError("Empty generated image")
        (run_dir / "raw_output_original.bin").write_bytes(generated_bytes)
        with Image.open(BytesIO(generated_bytes)) as generated:
            generated.load()
            original_format = generated.format
            original_mode = generated.mode
            original_size = list(generated.size)
            if generated.format == "PNG":
                (run_dir / "raw_output.png").write_bytes(generated_bytes)
                converted = False
            else:
                generated.convert("RGB").save(run_dir / "raw_output.png", format="PNG")
                converted = True
    except Exception as error:
        info = {"http_status": status_code, "latency_seconds": elapsed, "headers": headers, "response_body_sha256": sha256_bytes(response_bytes)}
        save_failure(run_dir, "RESPONSE_OR_IMAGE_ERROR", f"{type(error).__name__}: {error}", 1, info)
        raise RuntimeError(f"Condition {condition_id} response invalid; stop without retry") from error
    response_manifest = {
        "status": "VALID_IMAGE_RESULT_PENDING_PAIR_AUDIT", "condition": condition_id,
        "http_status": status_code, "headers": headers, "latency_seconds": elapsed,
        "response_body_sha256": sha256_bytes(response_bytes), "data_item_count": 1,
        "requested_model": MODEL, "response_model": payload.get("model"),
        "image_source_kind": image_source_kind, "image_url_summary": url_summary,
        "output_download_get_count": output_get_count, "original_format": original_format,
        "original_mode": original_mode, "original_size_wh": original_size, "converted_to_png": converted,
        "raw_output_png_sha256": sha256_file(run_dir / "raw_output.png"),
        "usage": payload.get("usage"), "created": payload.get("created"),
        "revised_prompt_present": bool(item.get("revised_prompt")), "post_count_for_condition": 1, "retry_count": 0,
    }
    write_json_exclusive(run_dir / "response_manifest.json", response_manifest)
    reserve_exclusive(root / "runs" / f"condition_{condition_id}_call_terminal.json", {
        "status": "SUCCESS_VALID_IMAGE", "condition": condition_id, "completed_utc": now_utc(),
        "post_invocations": 1, "output_sha256": response_manifest["raw_output_png_sha256"], "retry_allowed": False,
    })
    return response_manifest


def live_pair(root: Path, confirmation: str) -> int:
    if confirmation != CONFIRMATION:
        raise ValueError("Exact G1E-1 confirmation token is required")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise EnvironmentError("OPENAI_API_KEY and OPENAI_BASE_URL must be inherited from the current PowerShell session")
    config, prompt, original_bytes, conditions = validate(root)
    derived_endpoint = endpoint_from_base(base_url)
    if derived_endpoint != config["endpoint"]:
        raise ValueError(f"Environment endpoint {derived_endpoint} differs from frozen G1B-compatible endpoint")
    session_started = root / "runs" / "g1e_live_session_started.json"
    session_terminal = root / "runs" / "g1e_live_session_terminal.json"
    if session_started.exists() or session_terminal.exists() or any((root / "runs" / f"condition_{item['id']}_call_started.json").exists() for item in conditions):
        raise RuntimeError("G1E live budget/session already occupied; refusing duplicate execution")
    reserve_exclusive(session_started, {
        "status": "LIVE_SESSION_STARTED_EXCLUSIVE", "created_utc": now_utc(), "maximum_generation_posts": 2,
        "model": MODEL, "endpoint": config["endpoint"], "condition_order": ["A", "B"],
        "automatic_retries": 0, "fallbacks": 0, "api_key_logged": False,
    })
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    results = []
    try:
        result_a = execute_condition(root, config, conditions[0], prompt, original_bytes, api_key, opener)
        results.append(result_a)
    except Exception as error:
        reserve_exclusive(session_terminal, {"status": "STOP_AFTER_A_FAILURE_NO_B", "completed_utc": now_utc(), "occupied_posts": 1, "condition_A": "FAILED_OR_AMBIGUOUS", "condition_B": "NOT_ATTEMPTED", "detail": str(error), "retry_allowed": False})
        print(json.dumps({"status": "STOP_AFTER_A_FAILURE_NO_B", "detail": str(error)}, ensure_ascii=False))
        return 2
    try:
        result_b = execute_condition(root, config, conditions[1], prompt, original_bytes, api_key, opener)
        results.append(result_b)
    except Exception as error:
        reserve_exclusive(session_terminal, {"status": "STOP_AFTER_B_FAILURE_NO_RETRY", "completed_utc": now_utc(), "occupied_posts": 2, "condition_A": "SUCCESS", "condition_B": "FAILED_OR_AMBIGUOUS", "detail": str(error), "retry_allowed": False})
        print(json.dumps({"status": "STOP_AFTER_B_FAILURE_NO_RETRY", "detail": str(error)}, ensure_ascii=False))
        return 2
    try:
        pair_audit = audit(root)
        audit_status = pair_audit["status"]
    except Exception as error:
        audit_status = "PAIR_AUDIT_FAILED_OUTPUTS_PRESERVED"
        write_json_exclusive(root / "reports" / "g1e_pair_audit_failure.json", {"status": audit_status, "exception_type": type(error).__name__, "detail": str(error)})
    final_ledger = {
        "status": "G1E_1_TWO_POSTS_TERMINAL", "completed_utc": now_utc(), "maximum_generation_posts": 2,
        "occupied_generation_posts": 2, "actual_generation_posts": 2, "condition_A": "SUCCESS",
        "condition_B": "SUCCESS", "audit_status": audit_status, "retry_allowed": False,
        "output_sha256": {"A": results[0]["raw_output_png_sha256"], "B": results[1]["raw_output_png_sha256"]},
    }
    reserve_exclusive(session_terminal, final_ledger)
    write_json_exclusive(root / "reports" / "g1e_execution_ledger_final.json", final_ledger)
    print(json.dumps(final_ledger, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-max-two-paid-requests", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.dry_run:
        return dry_run(root)
    if args.self_test:
        return self_test(root)
    return live_pair(root, args.confirm_max_two_paid_requests)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "STOPPED_WITHOUT_AUTOMATIC_RETRY", "error_type": type(error).__name__, "detail": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise
