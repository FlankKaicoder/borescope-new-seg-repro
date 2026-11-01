#!/usr/bin/env python3
"""Perform non-billable GET/HEAD/OPTIONS capability probes; never POST to image endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse


ALLOWED_METHODS = {"GET", "HEAD", "OPTIONS"}


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def redact_headers(headers) -> dict:
    allowed = {"content-type", "allow", "server", "www-authenticate", "location", "x-request-id"}
    return {key.lower(): value for key, value in headers.items() if key.lower() in allowed}


def json_metadata(body: bytes, content_type: str) -> dict:
    metadata = {
        "parse_status": "NOT_JSON",
        "top_level_keys": [],
        "is_openapi_schema": False,
        "has_v1_images_edits_path": False,
        "has_mask_field": False,
    }
    if "json" not in content_type.lower() or not body:
        return metadata
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        metadata["parse_status"] = "INVALID_JSON"
        return metadata
    metadata["parse_status"] = "PASS"
    if isinstance(value, dict):
        metadata["top_level_keys"] = sorted(str(key) for key in value.keys())
        metadata["is_openapi_schema"] = "openapi" in value and isinstance(value.get("paths"), dict)
        paths = value.get("paths", {}) if metadata["is_openapi_schema"] else {}
        metadata["has_v1_images_edits_path"] = "/v1/images/edits" in paths
        serialized = json.dumps(value, ensure_ascii=False).lower()
        metadata["has_mask_field"] = '"mask"' in serialized
    return metadata


def probe(base_url: str, path: str, method: str, api_key: str | None, authenticate: bool = False) -> dict:
    if method not in ALLOWED_METHODS:
        raise ValueError(f"Forbidden probe method: {method}")
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    headers = {"User-Agent": "Exp13-G1B-GateA-DryRun/1.0", "Accept": "application/json,text/html;q=0.8,*/*;q=0.1"}
    if authenticate and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15, context=ssl.create_default_context()) as response:
            body = response.read(512 * 1024) if method == "GET" else b""
            content_type = response.headers.get("content-type", "")
            return {
                "path": path, "method": method, "url": url, "authenticated": bool(authenticate and api_key),
                "status": response.status, "headers": redact_headers(response.headers),
                "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes_captured": len(body),
                "body_preview": body[:1000].decode("utf-8", "replace"), "transport_error": None,
                "json_metadata": json_metadata(body, content_type),
            }
    except urllib.error.HTTPError as error:
        body = error.read(512 * 1024) if method == "GET" else b""
        content_type = error.headers.get("content-type", "")
        return {
            "path": path, "method": method, "url": url, "authenticated": bool(authenticate and api_key),
            "status": error.code, "headers": redact_headers(error.headers),
            "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes_captured": len(body),
            "body_preview": body[:1000].decode("utf-8", "replace"), "transport_error": None,
            "json_metadata": json_metadata(body, content_type),
        }
    except Exception as error:
        return {
            "path": path, "method": method, "url": url, "authenticated": bool(authenticate and api_key),
            "status": None, "headers": {}, "body_sha256": None, "body_bytes_captured": 0,
            "body_preview": "", "transport_error": f"{type(error).__name__}: {error}",
            "json_metadata": json_metadata(b"", ""),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    fallback = load_config(args.config)
    env_base = os.getenv("OPENAI_BASE_URL")
    env_model = os.getenv("OPENAI_IMAGE_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = env_base or fallback["base_url"]
    model = env_model or fallback["image_model"]
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("OPENAI_BASE_URL must be an absolute HTTPS URL")

    plan = [
        ("/", "GET", False),
        ("/docs", "GET", False),
        ("/openapi.json", "GET", False),
        ("/v1", "GET", False),
        ("/v1/models", "GET", True),
        ("/v1/images/edits", "OPTIONS", False),
        ("/v1/images/edits", "HEAD", False),
        ("/v1/images/generations", "OPTIONS", False),
    ]
    results = [probe(base_url, path, method, api_key, authenticated) for path, method, authenticated in plan]
    joined = "\n".join((item.get("body_preview") or "") for item in results).lower()
    openapi_candidate = next((item for item in results if item["path"] == "/openapi.json" and item["status"] == 200), None)
    openapi = openapi_candidate if openapi_candidate and openapi_candidate["json_metadata"]["is_openapi_schema"] else None
    openapi_has_edit = bool(openapi and openapi["json_metadata"]["has_v1_images_edits_path"])
    openapi_has_mask = bool(openapi and openapi["json_metadata"]["has_mask_field"])
    v1_style_evidence = any(item["path"].startswith("/v1") and item["status"] in {200, 400, 401, 403, 405, 422} for item in results)
    edit_route_evidence = any(item["path"] == "/v1/images/edits" and item["status"] in {200, 400, 401, 403, 405, 415, 422} for item in results)
    docs_claim_images = "images" in joined or "image" in joined
    capability = "API_CAPABILITY_UNVERIFIED"
    mask_status = "API_CAPABILITY_UNVERIFIED"
    if openapi_has_edit and openapi_has_mask:
        capability = "IMAGE_EDIT_SCHEMA_ADVERTISED_NOT_LIVE_TESTED"
        mask_status = "MASK_EDIT_SCHEMA_ADVERTISED_NOT_LIVE_TESTED"
    elif edit_route_evidence:
        capability = "IMAGE_EDIT_ROUTE_POSSIBLE_NOT_CONFIRMED"
    if openapi_has_edit and not openapi_has_mask:
        mask_status = "MASK_EDIT_UNSUPPORTED"

    report = {
        "status": capability,
        "mask_edit_status": mask_status,
        "probe_time_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "openai_api_key_present": bool(api_key),
            "openai_base_url": base_url,
            "openai_image_model": model,
            "base_url_source": "environment" if env_base else fallback["provenance"],
            "model_source": "environment" if env_model else fallback["provenance"],
            "api_key_value_logged": False,
        },
        "safety": {
            "allowed_methods": sorted(ALLOWED_METHODS),
            "post_requests": 0,
            "image_generation_requests": 0,
            "automatic_retries": 0,
            "model_switches": 0,
        },
        "evidence": {
            "v1_path_style_possible": v1_style_evidence,
            "image_edit_route_possible": edit_route_evidence,
            "site_or_docs_mentions_images": docs_claim_images,
            "openapi_schema_retrieved": bool(openapi),
            "openapi_has_images_edits": openapi_has_edit,
            "openapi_has_mask_field": openapi_has_mask,
            "model_availability_confirmed": model in joined,
        },
        "requests": results,
        "limitations": [
            "No POST or multipart request was sent, so operational image editing cannot be confirmed.",
            "An endpoint-like HTTP status is not proof that the configured model accepts image+mask+prompt.",
            "Independent mask support requires provider documentation/schema or an explicitly authorized Gate B request.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": capability, "mask_edit_status": mask_status, "post_requests": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
