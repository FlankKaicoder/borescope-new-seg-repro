"""Run the frozen Exp14 request plan with the G3 transport and permanent request locks."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.prepare_large_scale_repair import (
    FINAL_SHA, PHASE, PRIVATE_ROOT, REPO, canonical_digest, hash_file, load_json,
    request_identity, target_mask, verify_mask,
)

STOP_HTTP = {401, 402, 403, 404, 429}


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen_plan(root):
    auth_path = root / "authorization.json"
    auth = load_json(auth_path)
    g3.require(auth["record_type"] == "EXP14_LARGE_SCALE_REPAIR_EXPLICIT_USER_AUTHORIZATION" and
               auth["source_csv_sha256"] == FINAL_SHA and auth["n"] == 1, "Wrong Exp14 authorization")
    hash_file(root / "authorization_task.txt", auth["authorization_attachment_sha256"])
    hash_file(root / "scope.json", auth["scope_sha256"])
    hash_file(root / "request_manifest.csv", auth["request_manifest_sha256"])
    protocol, protocol_sha = g3.load_protocol(REPO)
    g3.require(auth["protocol_sha256"] == protocol_sha, "Frozen G2 protocol mismatch")
    scope = load_json(root / "scope.json")
    request = protocol["request"]
    rows = scope["requests"]
    ids = [row["request_id"] for row in rows]
    g3.require(len(ids) == len(set(ids)) == auth["max_posts"] and ids == auth["request_ids"],
               "Request scope exceeds or differs from the authorized exact set")
    for row in rows:
        source = scope["sources"][row["source_id"]]
        g3.require(source["split"] == "train" and source["decision"] == "SELECT" and not source.get("hold") and
                   row["source_image_sha256"] == source["image_sha256"] and
                   row["source_json_sha256"] == source["annotation_sha256"] and
                   row["selected_polygon_ids"].split("|") == source["selected_polygon_ids"] and
                   row["request_id"] == request_identity(row) and row["model"] == request["model"] and
                   row["endpoint"] == request["base_url"] + request["endpoint"] and row["n"] == "1",
                   "Invalid request/source binding")
    return scope, hash_file(auth_path), protocol_sha


def input_for_row(root, row, source):
    annotation = Path(source["annotation"])
    hash_file(annotation, source["annotation_sha256"])
    target = target_mask(source, load_json(annotation)["shapes"])
    g3.require(g3.digest(target.tobytes()) == row["target_alpha_sha256"], "Selected Polygon raster changed")
    mask = g3.safe_file(root, row["mask_path"])
    verify_mask(mask, target, row["mask_sha256"])
    return g3.validate_input(row, root, root if row["prompt_root"] == "run" else REPO)


def save_returned_image(folder, raw, response, requested_model, elapsed, status, secret=None):
    ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}[response["image"]["format"]]
    raw_name = f"raw_output.{ext}"
    g3.write_exclusive(folder / raw_name, raw)
    returned = response.get("response_model")
    response.update({
        "raw_filename": raw_name, "http_status": status, "post_invocations": 1,
        "elapsed_seconds": round(elapsed, 3),
        "cost_status": "COST_UNKNOWN" if response["reported_cost"] is None else "PROVIDER_REPORTED",
        "requested_model": requested_model,
        "model_metadata": "UNREPORTED" if returned is None else "MATCH" if returned == requested_model else "DIFFERENT",
    })
    data = json.dumps(response, ensure_ascii=False, indent=2).encode("utf-8")
    g3.write_exclusive(folder / "response_manifest.json", data.replace(secret, b"[REDACTED]") if secret else data)


def run_one(root, row, source, credential, transport, authorization_sha, protocol_sha,
            input_validator=None, historical_locks=None):
    request_id = row["request_id"]
    folder = root / "live" / row["variant_id"]
    lock = root / "locks" / f"{request_id}.json"
    if lock.exists() or folder.exists() or (historical_locks and (historical_locks / lock.name).exists()):
        terminal = folder / "TERMINAL.json"
        return {"status": "LOCKED_NO_POST", "terminal": load_json(terminal)["status"] if terminal.is_file() else None}

    inp = (input_validator or input_for_row)(root, row, source)
    body, content_type = g3.encode_multipart(
        row["source_id"], row["model"], inp["prompt_wire"], inp["image_path"].name,
        inp["image_bytes"], inp["image_format"], inp["mask_bytes"],
    )
    request = {**row, "method": "POST", "body_sha256": g3.digest(body), "body_bytes": len(body),
               "content_type": content_type, "authorization_sha256": authorization_sha,
               "protocol_sha256": protocol_sha, "automatic_retries": 0, "redirects": False}
    try:
        g3.reserve_lock(lock, {"request_id": request_id, "variant_id": row["variant_id"], "created_utc": now(),
                               "request_sha256": canonical_digest(request),
                               "status": "PERMANENT_SLOT_CONSUMED_BEFORE_POST"})
    except FileExistsError:
        return {"status": "LOCKED_NO_POST"}

    folder.mkdir(parents=True, exist_ok=False)
    g3.write_json_exclusive(folder / "request_manifest.json", request)
    g3.write_json_exclusive(folder / "POST_STARTED.json", {"created_utc": now(), "post_invocations": 1})
    start = time.monotonic()
    status = None
    terminal = {"status": "FAILED_NO_RETRY", "post_invocations": 1, "cost_status": "COST_UNKNOWN"}
    secret = credential.encode("utf-8")
    try:
        status, headers, response_body = transport.post(row["endpoint"], {
            "Authorization": "Bearer " + credential, "Content-Type": content_type,
            "Content-Length": str(len(body)), "Accept": "application/json,image/*",
        }, body)
        safe_headers = {key: value.replace(credential, "[REDACTED]") for key, value in
                        g3.safe_header_summary(headers).items()}
        g3.write_json_exclusive(folder / "http_response.json", {
            "http_status": status, "headers": safe_headers, "response_body_sha256": g3.digest(response_body),
            "bytes": len(response_body), "body_truncated": len(response_body) > g3.MAX_RESPONSE_BYTES,
        })
        g3.write_exclusive(folder / "response_body.bin", response_body.replace(secret, b"[REDACTED]"))
        raw, response = g3.parse_output(status, headers, response_body, transport)
        save_returned_image(folder, raw, response, row["model"], time.monotonic() - start, status, secret)
        terminal.update(status="RETURNED_IMAGE", cost_status=response["cost_status"],
                        raw_sha256=response["raw_output_sha256"])
    except Exception as error:
        terminal.update(error_type=type(error).__name__, error=str(error).replace(credential, "[REDACTED]")[:300])
    terminal.update(http_status=status, finished_utc=now(), elapsed_seconds=round(time.monotonic() - start, 3),
                    delivery_uncertain=status is None or status >= 500)
    g3.write_json_exclusive(folder / "TERMINAL.json", terminal)
    return terminal


def recover_responses(root, scope=None, inspected=None):
    if scope is None:
        scope, _, _ = frozen_plan(root)
    inspected = set() if inspected is None else inspected
    for row in scope["requests"]:
        if row["variant_id"] in inspected:
            continue
        folder = root / "live" / row["variant_id"]
        terminal_path = folder / "TERMINAL.json"
        if not terminal_path.is_file():
            continue
        try:
            terminal = load_json(terminal_path)
        except json.JSONDecodeError:
            continue
        inspected.add(row["variant_id"])
        error = terminal.get("error", "")
        if error not in {"Unexpected returned model", "Expected exactly one output image; source slot consumed"}:
            continue
        dest = root / "response_recovery" / row["variant_id"]
        if (dest / "RECOVERED.json").is_file():
            continue
        http = load_json(folder / "http_response.json")
        body = folder / "response_body.bin"
        hash_file(body, http["response_body_sha256"])
        original = body.read_bytes()
        if error == "Unexpected returned model":
            payloads = [original]
        else:
            original_json = json.loads(original)
            items = original_json.get("data")
            if not isinstance(items, list) or not 2 <= len(items) <= 8:
                continue
            payloads = [json.dumps({**original_json, "data": [item]}).encode("utf-8") for item in items]
        outputs = []
        for index, payload in enumerate(payloads, start=1):
            output = dest if len(payloads) == 1 else dest / f"output_{index}"
            try:
                raw, response = g3.parse_output(http["http_status"], http["headers"], payload, g3.HttpTransport())
            except (ValueError, OSError) as exc:
                g3.write_json_exclusive(root / "logs" / f"response_recovery_error_{time.time_ns()}.json",
                                        {"variant_id": row["variant_id"], "index": index, "error_type": type(exc).__name__})
                continue
            output.mkdir(parents=True, exist_ok=False)
            save_returned_image(output, raw, response, row["model"], terminal["elapsed_seconds"], http["http_status"])
            outputs.append({"position": index, "path": output.relative_to(dest).as_posix(),
                            "raw_sha256": response["raw_output_sha256"]})
        g3.write_json_exclusive(dest / "RECOVERED.json", {
            "request_id": row["request_id"], "original_terminal_sha256": hash_file(terminal_path),
            "response_body_sha256": http["response_body_sha256"], "outputs": outputs,
            "raw_sha256": outputs[0]["raw_sha256"] if len(payloads) == len(outputs) == 1 else None,
            "provider_returned_positions": len(payloads),
            "api_posts": 0,
        })
        print(f"{row['variant_id']} recovered {len(outputs)}/{len(payloads)} images from persisted relay response", flush=True)


def execute(root, credential, workers, transport=None, *, plan_loader=frozen_plan,
            input_validator=input_for_row, historical_locks=None, log_dir=None, require_first_image=False):
    scope, auth_sha, protocol_sha = plan_loader(root)
    g3.require(bool(credential.strip()), "Missing Exp14 image API credential")
    transport = transport or g3.HttpTransport()
    rows = scope["requests"]
    pending = iter(rows)
    logs = log_dir if log_dir is not None else root / "logs"
    g3.write_json_exclusive(logs / f"start_{time.time_ns()}.json",
                           {"started_utc": now(), "authorized_requests": len(rows), "max_in_flight": workers})
    counts = Counter()
    transient = 0
    stop_reason = None

    def handle(future, row):
        nonlocal transient, stop_reason
        try:
            result = future.result()
        except Exception as error:
            # A local write failure can leave a lock with ambiguous delivery.
            stop_reason = stop_reason or "LOCAL_IO_OR_INPUT_ERROR"
            g3.write_json_exclusive(logs / f"error_{time.time_ns()}.json",
                                    {"variant_id": row["variant_id"], "error_type": type(error).__name__,
                                     "error": str(error).replace(credential, "[REDACTED]")[:300]})
            counts["LOCAL_IO_OR_INPUT_ERROR"] += 1
            return
        state, status = result["status"], result.get("http_status")
        counts[state] += 1
        if state != "LOCKED_NO_POST":
            if state == "RETURNED_IMAGE" or status in {400, 413, 415, 422}:
                transient = 0
            else:
                transient += 1
            if status in STOP_HTTP:
                stop_reason = stop_reason or f"RELAY_HTTP_{status}"
            if transient >= 3:
                stop_reason = stop_reason or "THREE_CONSECUTIVE_RELAY_OR_TRANSPORT_FAILURES"
            print(f"{row['variant_id']} {state} HTTP={status}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def submit():
            row = next(pending, None)
            if row is not None:
                source = scope["sources"][row["source_id"]]
                future = pool.submit(run_one, root, row, source, credential, transport, auth_sha, protocol_sha,
                                     input_validator, historical_locks)
                active[future] = row

        # Check one real request before raising concurrency. It counts in the frozen plan.
        submit()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                handle(future, active.pop(future))
            if not stop_reason:
                concurrency = 1 if require_first_image and not counts["RETURNED_IMAGE"] else workers
                for _ in range(concurrency - len(active)):
                    submit()

    receipt = {"finished_utc": now(), "counts_this_process": dict(counts), "stop_reason": stop_reason}
    g3.write_json_exclusive(logs / f"stop_{time.time_ns()}.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--recover-responses", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    root = (PRIVATE_ROOT / PHASE).resolve()
    if args.live:
        credential = os.environ.pop("EXP14_IMAGE_API_KEY", "")
        execute(root, credential, args.workers)
    elif args.recover_responses:
        recover_responses(root)
    else:
        scope, _, _ = frozen_plan(root)
        for row in scope["requests"]:
            input_for_row(root, row, scope["sources"][row["source_id"]])
        print(f"OFFLINE_PREFLIGHT_PASS {len(scope['requests'])} requests, POST=0")


if __name__ == "__main__":
    main()
