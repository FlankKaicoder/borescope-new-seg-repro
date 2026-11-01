#!/usr/bin/env python3
"""Issue at most one corrosion23 image-edit POST, then save and audit the result."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from audit_live_output import audit_run


MODEL = "gpt-image-2.5-sunburst"
SOURCE = "corrosion23"
PROMPT_ID = "CLOUD_RESTORATION_V1"
CONFIRMATION = "AUTHORIZE_ONE_CORROSION23_POST"
STATUS = "FIRST_CALL_RAW / PENDING_HUMAN_REVIEW"


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


def multipart_body(fields: list[tuple[str, str]], files: list[tuple[str, str, str, bytes]]) -> tuple[bytes, str]:
    boundary = "----Exp13G1B" + uuid.uuid4().hex
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


def safe_headers(headers) -> dict:
    allowed = {"content-type", "content-length", "x-request-id", "openai-request-id", "server", "date"}
    return {key.lower(): value for key, value in headers.items() if key.lower() in allowed}


def allocate_run_dir(root: Path) -> Path:
    preferred = root / "runs" / "corrosion23_first_call_live"
    candidates = [preferred]
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidates.append(root / "runs" / f"corrosion23_first_call_live_{stamp}")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("Could not allocate a unique live run directory")


def validate_inputs(root: Path, config: dict) -> dict:
    gate_a = root / "runs" / "corrosion23_dry_run"
    paths = {
        "original.png": gate_a / "original.png",
        "original_annotation.json": gate_a / "original_annotation.json",
        "polygon_mask.png": gate_a / "polygon_mask.png",
        "api_mask.png": gate_a / "api_mask.png",
        "overlay_check.png": gate_a / "overlay_check.png",
        "prepare_report.json": gate_a / "prepare_report.json",
        "prompt.txt": gate_a / "prompt.txt",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    checks = {
        "original.png": config["original_sha256"],
        "polygon_mask.png": config["polygon_mask_sha256"],
        "api_mask.png": config["api_mask_sha256"],
        "prompt.txt": config["prompt_sha256"],
    }
    for name, expected in checks.items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise ValueError(f"Frozen input hash mismatch for {name}: {actual}")
    with Image.open(paths["original.png"]) as image:
        image.load()
        image_size = image.size
        if image.format != "PNG":
            raise ValueError("Original input must be PNG")
    with Image.open(paths["api_mask.png"]) as mask:
        mask.load()
        if mask.format != "PNG" or mask.mode != "RGBA" or mask.size != image_size:
            raise ValueError("API mask must be same-size RGBA PNG")
        if mask.getchannel("A").getextrema() != (0, 255):
            raise ValueError("API mask must contain transparent edit and opaque preserve pixels")
    return {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in paths.items()}


def save_failure(run_dir: Path, request_manifest: dict, category: str, detail: str, response_info: dict | None = None) -> None:
    response = {
        "status": "FIRST_CALL_FAILED / STOP_NO_RETRY",
        "failure_category": category,
        "detail": detail,
        "post_request_count": request_manifest.get("post_request_count", 0),
        "response": response_info,
        "retry_allowed": False,
        "billing_status": "UNKNOWN_DO_NOT_RETRY" if request_manifest.get("post_request_count") else "NOT_SENT",
    }
    response_path = run_dir / "response_manifest.json"
    if not response_path.exists():
        write_json_exclusive(response_path, response)
    audit_path = run_dir / "audit_report.json"
    if not audit_path.exists():
        write_json_exclusive(audit_path, {
            "status": "FIRST_CALL_FAILED / NO_IMAGE / PENDING_HUMAN_REVIEW",
            "source": SOURCE,
            "model": MODEL,
            "image_result_received": False,
            "technical_audit": "NOT_AVAILABLE_NO_IMAGE",
            "training_pair_eligibility": "NOT_EVALUATED_NOT_FROZEN",
        })
    summary_path = run_dir / "first_call_summary.md"
    if not summary_path.exists():
        summary_path.write_text(
            "# Exp13-G1B corrosion23 first live call\n\n"
            "Status: `FIRST_CALL_FAILED / STOP_NO_RETRY`\n\n"
            f"- Model: `{MODEL}`\n"
            f"- Failure category: `{category}`\n"
            f"- POST invocation count: `{request_manifest.get('post_request_count', 0)}`\n"
            "- Image result: `NO`\n"
            "- Billing status: `UNKNOWN_DO_NOT_RETRY`\n\n"
            "No retry, fallback, or second request is authorized.\n",
            encoding="utf-8",
        )


def preflight(root: Path, config_path: Path, report_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = validate_inputs(root, config)
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL")
    report = {
        "status": "READY" if api_key_present and base_url else "READY_EXCEPT_ENVIRONMENT_INHERITANCE",
        "source": SOURCE,
        "forced_model": MODEL,
        "api_key_present": api_key_present,
        "base_url_present": bool(base_url),
        "environment_image_model_ignored": True,
        "inputs": inputs,
        "maximum_generation_requests": 1,
        "automatic_retries": 0,
        "model_fallbacks": 0,
        "live_request_sent": False,
    }
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    write_json_exclusive(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def live_once(root: Path, config_path: Path, confirmation: str) -> int:
    if confirmation != CONFIRMATION:
        raise ValueError("Missing exact one-call confirmation token")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise EnvironmentError("OPENAI_API_KEY and OPENAI_BASE_URL must be inherited from the current PowerShell session")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["model"] != MODEL or config["maximum_generation_requests"] != 1 or config["n"] != 1:
        raise ValueError("Gate B frozen config mismatch")
    inputs = validate_inputs(root, config)
    ledger_started = root / "runs" / "gate_b_corrosion23_call_ledger_started.json"
    ledger_terminal = root / "runs" / "gate_b_corrosion23_call_ledger_terminal.json"
    if ledger_started.exists() or ledger_terminal.exists():
        raise RuntimeError("Gate B one-call budget is already occupied; refusing another request")
    run_dir = allocate_run_dir(root)
    for name, info in inputs.items():
        shutil.copy2(info["path"], run_dir / name)
        if sha256_file(run_dir / name) != info["sha256"]:
            raise ValueError(f"Copy verification failed for {name}")

    endpoint = endpoint_from_base(base_url)
    prompt = (run_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    image_bytes = (run_dir / "original.png").read_bytes()
    mask_bytes = (run_dir / "api_mask.png").read_bytes()
    body, content_type = multipart_body(
        [("model", MODEL), ("prompt", prompt), ("n", "1")],
        [("image[]", "original.png", "image/png", image_bytes), ("mask", "api_mask.png", "image/png", mask_bytes)],
    )
    request_manifest = {
        "status": "FROZEN_BEFORE_POST",
        "created_utc": now_utc(),
        "source": SOURCE,
        "prompt_id": PROMPT_ID,
        "model": MODEL,
        "environment_image_model_ignored": os.getenv("OPENAI_IMAGE_MODEL") != MODEL,
        "endpoint": endpoint,
        "method": "POST",
        "content_type": "multipart/form-data",
        "fields": ["model", "prompt", "n", "image[]", "mask"],
        "n": 1,
        "prompt_sha256": sha256_file(run_dir / "prompt.txt"),
        "image_sha256": sha256_file(run_dir / "original.png"),
        "mask_sha256": sha256_file(run_dir / "api_mask.png"),
        "request_body_sha256": sha256_bytes(body),
        "request_body_bytes": len(body),
        "authorization_header_present": True,
        "api_key_logged": False,
        "automatic_retries": 0,
        "model_fallbacks": 0,
        "post_request_count": 1,
    }
    write_json_exclusive(run_dir / "request_manifest.json", request_manifest)
    write_json_exclusive(ledger_started, {
        "status": "STARTED_ONE_CALL_BUDGET_OCCUPIED",
        "created_utc": now_utc(),
        "source": SOURCE,
        "model": MODEL,
        "maximum_generation_requests": 1,
        "occupied": 1,
        "run_dir": str(run_dir.resolve()),
        "request_body_sha256": request_manifest["request_body_sha256"],
        "retry_allowed": False,
    })

    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "Accept": "application/json",
            "User-Agent": "Exp13-G1B-OneShot/1.0",
        },
    )
    opener = urllib.request.build_opener(
        NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context())
    )
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=600) as response:
            response_bytes = response.read()
            status_code = response.status
            headers = safe_headers(response.headers)
    except urllib.error.HTTPError as error:
        elapsed = time.perf_counter() - started
        error_body = error.read()
        (run_dir / "response_error_body.bin").write_bytes(error_body)
        info = {"http_status": error.code, "latency_seconds": elapsed, "headers": safe_headers(error.headers), "body_sha256": sha256_bytes(error_body), "body_bytes": len(error_body)}
        save_failure(run_dir, request_manifest, "HTTP_ERROR", f"HTTP {error.code}", info)
        write_json_exclusive(ledger_terminal, {"status": "FAILED_STOP_NO_RETRY", "post_invocations": 1, "run_dir": str(run_dir.resolve()), "response": info, "billing_status": "UNKNOWN_DO_NOT_RETRY"})
        print(json.dumps({"status": "FAILED_STOP_NO_RETRY", "run_dir": str(run_dir), "http_status": error.code}, ensure_ascii=False))
        return 2
    except Exception as error:
        elapsed = time.perf_counter() - started
        info = {"http_status": None, "latency_seconds": elapsed, "exception_type": type(error).__name__}
        save_failure(run_dir, request_manifest, "TRANSPORT_ERROR", f"{type(error).__name__}: {error}", info)
        write_json_exclusive(ledger_terminal, {"status": "FAILED_STOP_NO_RETRY", "post_invocations": 1, "run_dir": str(run_dir.resolve()), "response": info, "billing_status": "UNKNOWN_DO_NOT_RETRY"})
        print(json.dumps({"status": "FAILED_STOP_NO_RETRY", "run_dir": str(run_dir), "error": type(error).__name__}, ensure_ascii=False))
        return 2
    elapsed = time.perf_counter() - started
    (run_dir / "response_raw.json").write_bytes(response_bytes)
    if status_code not in {200, 201}:
        info = {"http_status": status_code, "latency_seconds": elapsed, "headers": headers, "body_sha256": sha256_bytes(response_bytes)}
        save_failure(run_dir, request_manifest, "UNEXPECTED_HTTP_STATUS", f"HTTP {status_code}", info)
        write_json_exclusive(ledger_terminal, {"status": "FAILED_STOP_NO_RETRY", "post_invocations": 1, "run_dir": str(run_dir.resolve()), "response": info, "billing_status": "UNKNOWN_DO_NOT_RETRY"})
        return 2
    try:
        payload = json.loads(response_bytes.decode("utf-8"))
        data = payload["data"]
        if not isinstance(data, list) or len(data) != 1:
            raise ValueError(f"Expected exactly one data item, got {len(data) if isinstance(data, list) else type(data).__name__}")
        item = data[0]
        source_kind = None
        source_url_summary = None
        output_request_count = 0
        if item.get("b64_json"):
            generated_bytes = base64.b64decode(item["b64_json"], validate=True)
            source_kind = "b64_json"
        elif item.get("url"):
            source_kind = "url"
            parsed_url = urlparse(item["url"])
            source_url_summary = {"scheme": parsed_url.scheme, "host": parsed_url.netloc, "url_sha256": sha256_bytes(item["url"].encode("utf-8"))}
            if parsed_url.scheme != "https" or not parsed_url.netloc:
                raise ValueError("Returned image URL is not absolute HTTPS")
            output_request_count = 1
            image_request = urllib.request.Request(item["url"], method="GET", headers={"User-Agent": "Exp13-G1B-OneShot/1.0"})
            with opener.open(image_request, timeout=120) as image_response:
                generated_bytes = image_response.read()
        else:
            raise ValueError("Response data item contains neither b64_json nor url")
        if not generated_bytes:
            raise ValueError("Decoded output image is empty")
        original_payload_path = run_dir / "raw_output_original.bin"
        original_payload_path.write_bytes(generated_bytes)
        with Image.open(BytesIO(generated_bytes)) as generated:
            generated.load()
            detected_format = generated.format
            detected_mode = generated.mode
            detected_size = list(generated.size)
            if detected_format == "PNG":
                (run_dir / "raw_output.png").write_bytes(generated_bytes)
                converted_to_png = False
            else:
                generated.convert("RGB").save(run_dir / "raw_output.png", format="PNG")
                converted_to_png = True
        (run_dir / "raw_output_sha256.txt").write_text(f"{sha256_file(run_dir / 'raw_output.png')}  raw_output.png\n", encoding="ascii")
    except Exception as error:
        info = {"http_status": status_code, "latency_seconds": elapsed, "headers": headers, "response_body_sha256": sha256_bytes(response_bytes)}
        save_failure(run_dir, request_manifest, "RESPONSE_PARSE_OR_IMAGE_ERROR", f"{type(error).__name__}: {error}", info)
        write_json_exclusive(ledger_terminal, {"status": "FAILED_STOP_NO_RETRY", "post_invocations": 1, "run_dir": str(run_dir.resolve()), "response": info, "billing_status": "UNKNOWN_DO_NOT_RETRY"})
        print(json.dumps({"status": "FAILED_STOP_NO_RETRY", "run_dir": str(run_dir), "error": type(error).__name__}, ensure_ascii=False))
        return 2

    response_manifest = {
        "status": STATUS,
        "requested_model": MODEL,
        "response_model": payload.get("model"),
        "response_model_matches_requested": None if payload.get("model") is None else payload.get("model") == MODEL,
        "http_status": status_code,
        "headers": headers,
        "latency_seconds": elapsed,
        "response_body_sha256": sha256_bytes(response_bytes),
        "response_body_bytes": len(response_bytes),
        "data_item_count": 1,
        "image_source_kind": source_kind,
        "image_url_summary": source_url_summary,
        "output_download_get_count": output_request_count,
        "detected_original_format": detected_format,
        "detected_original_mode": detected_mode,
        "detected_size_wh": detected_size,
        "converted_to_png": converted_to_png,
        "raw_output_original_sha256": sha256_file(original_payload_path),
        "raw_output_png_sha256": sha256_file(run_dir / "raw_output.png"),
        "usage": payload.get("usage"),
        "created": payload.get("created"),
        "revised_prompt_present": bool(item.get("revised_prompt")),
        "post_request_count": 1,
        "retry_count": 0,
    }
    write_json_exclusive(run_dir / "response_manifest.json", response_manifest)
    try:
        audit = audit_run(run_dir)
        terminal_status = "SUCCESS_FIRST_CALL_RAW_PENDING_HUMAN_REVIEW"
    except Exception as error:
        terminal_status = "SUCCESS_IMAGE_SAVED_AUDIT_FAILED_NO_RETRY_NEEDED"
        write_json_exclusive(run_dir / "audit_failure.json", {"status": terminal_status, "exception_type": type(error).__name__, "detail": str(error), "generation_retry_allowed": False})
        audit = None
    write_json_exclusive(ledger_terminal, {
        "status": terminal_status,
        "completed_utc": now_utc(),
        "post_invocations": 1,
        "successful_image_results": 1,
        "run_dir": str(run_dir.resolve()),
        "model": MODEL,
        "raw_output_png_sha256": response_manifest["raw_output_png_sha256"],
        "retry_allowed": False,
    })
    print(json.dumps({
        "status": terminal_status,
        "result_status": STATUS,
        "run_dir": str(run_dir.resolve()),
        "output_sha256": response_manifest["raw_output_png_sha256"],
        "audit_completed": audit is not None,
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-single-paid-request", default="")
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "configs" / "gate_b_config.json"
    if args.preflight:
        report = args.preflight_report or root / "reports" / "gateB_preflight.json"
        return preflight(root, config_path, report)
    return live_once(root, config_path, args.confirm_single_paid_request)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "STOPPED_BEFORE_OR_DURING_ONE_SHOT", "error_type": type(error).__name__, "detail": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise
