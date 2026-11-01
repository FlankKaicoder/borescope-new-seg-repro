"""Default offline. One future POST only after receipt + explicit confirmation.

This CLI's live branch must not be executed during G1F-1A.
"""
import argparse
import base64
import http.client
import io
import ipaddress
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import (ROOT, BASE, MODEL, ENDPOINT, require, validate_inputs, request_metadata,
                      build_multipart, validate_bundle, validate_approval, create_started_marker,
                      write_json, digest, file_sha)


def utc():
    return datetime.now(timezone.utc).isoformat()


def offline_guard(event, args):
    if event.startswith(('socket.', 'urllib.', 'http.client.', 'subprocess.')) or event == 'os.system':
        raise RuntimeError('Network/subprocess blocked in dry run')


def sanitized_json(value, key):
    """Do not persist returned credential echoes or signed image URLs."""
    if isinstance(value, dict):
        return {k: ('[IMAGE_DATA_OMITTED]' if k == 'b64_json' else
                    url_summary(v) if k == 'url' and isinstance(v, str) else sanitized_json(v, key))
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitized_json(v, key) for v in value]
    if isinstance(value, str):
        return value.replace(key, '[REDACTED_CREDENTIAL]') if key else value
    return value


def url_summary(value):
    parsed = urlsplit(value)
    return {'scheme': parsed.scheme, 'host': parsed.hostname, 'path_sha256': digest(parsed.path.encode()),
            'has_query': bool(parsed.query), 'full_url_recorded': False}


def download_once(url, run):
    # Output retrieval is one GET, never a second generation request. No auth forwarding.
    parsed = urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname and not parsed.username
            and not parsed.password and parsed.port in (None, 443), 'Unsupported output URL')
    require(parsed.hostname.lower() not in ('localhost',) and not parsed.hostname.lower().endswith('.local'),
            'Local output host rejected')
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    require(address is None or address.is_global, 'Private output IP rejected')
    write_json(run / 'output_download_started.json', {'utc': utc(), 'method': 'GET',
               'url': url_summary(url), 'redirects': False, 'retries': 0, 'authorization_forwarded': False})
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=120)
    try:
        path = parsed.path or '/'
        if parsed.query:
            path += '?' + parsed.query
        connection.request('GET', path, headers={'Accept': 'image/*'})
        response = connection.getresponse()
        require(response.status == 200, 'Output download failed or redirected: STOP')
        image_bytes = response.read(100 * 1024 * 1024 + 1)
        require(len(image_bytes) <= 100 * 1024 * 1024, 'Output exceeds local size limit')
        return image_bytes
    finally:
        connection.close()


def dry_run(output):
    sys.addaudithook(offline_guard)
    validate_bundle()
    checks = validate_inputs()
    metadata = request_metadata()
    secret = os.environ.pop('OPENAI_API_KEY', None)
    credential_present = bool(secret and secret.strip())
    secret = None
    require(credential_present, 'DPAPI child credential not present')
    require(os.environ.get('OPENAI_BASE_URL') == BASE and os.environ.get('OPENAI_IMAGE_MODEL') == MODEL,
            'Unexpected loaded configuration')
    require(not (ROOT / 'STARTED.json').exists(), 'Production budget already occupied')
    output = output.resolve()
    require(output.parent == ROOT / 'dry_runs' and not output.exists(), 'Invalid dry-run output directory')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'DRY_RUN_PASS', 'checks': checks, 'request': metadata,
              'credential_present_in_child': credential_present, 'credential_removed_from_child_environment': True,
              'key_logged': False, 'network_guard': True, 'api_post_count': 0, 'network_requests': 0,
              'production_started_marker_exists': False, 'live_executed': False,
              'approval_gate': 'HOLD_PENDING_INDEPENDENT_CHATGPT_GATE', 'utc': utc()}
    write_json(output / 'dry_run_report.json', report)
    print('DRY_RUN_PASS; API_POST=0')


def live(confirmation, receipt):
    # No callers in preflight. Validation is fail-closed before marker/network access.
    validate_bundle()
    validate_inputs()
    validate_approval(receipt.resolve(), confirmation)
    require(not (ROOT / 'STARTED.json').exists(), 'Permanent budget marker exists: never resend')
    key = os.environ.pop('OPENAI_API_KEY', None)
    require(bool(key and key.strip()), 'Missing DPAPI child credential')
    require(os.environ.get('OPENAI_BASE_URL') == BASE and os.environ.get('OPENAI_IMAGE_MODEL') == MODEL,
            'Configuration mismatch')
    body, content_type = build_multipart()
    frozen_request = json.loads((ROOT / 'request_manifest.json').read_text(encoding='utf-8'))
    require(digest(body) == frozen_request['body_sha256']
            and len(body) == frozen_request['body_bytes'], 'In-memory request differs from frozen request: STOP')
    run = ROOT / 'live_result'
    # Refuse re-use even if a marker was externally removed; do not offer a reset option.
    require(not run.exists(), 'Live result path already exists: STOP')
    run.mkdir(exist_ok=False)
    write_json(run / 'request_manifest.json', {**request_metadata(), 'utc': utc(),
               'approval_sha256': file_sha(receipt), 'status': 'FROZEN_BEFORE_POST'})
    create_started_marker(ROOT / 'STARTED.json', {
        'status': 'PERMANENT_BUDGET_OCCUPIED', 'utc': utc(), 'max_posts': 1,
        'source': 'corrosion2', 'model': MODEL, 'body_sha256': digest(body),
        'bundle_lock_sha256': file_sha(ROOT / 'bundle_lock.json'),
        'rule': 'Never delete/reuse; ambiguous failure consumes the entire budget.'})
    connection = None
    start = time.monotonic()
    post_invoked = False
    try:
        connection = http.client.HTTPSConnection('direct-api.ccccapi.cc', timeout=600)
        # Exactly one call site; http.client has no redirect following or automatic retry.
        post_invoked = True
        connection.request('POST', '/v1/images/edits', body=body,
                           headers={'Authorization': 'Bearer ' + key, 'Content-Type': content_type,
                                    'Content-Length': str(len(body)), 'Accept': 'application/json'})
        response = connection.getresponse()
        status = response.status
        data = response.read(100 * 1024 * 1024 + 1)
        elapsed = time.monotonic() - start
        safe_headers = {name: response.getheader(name) for name in
                        ('content-type', 'content-length', 'x-request-id', 'request-id')
                        if response.getheader(name) is not None}
        write_json(run / 'http_response_metadata.json', {
            'http_status': status, 'headers': sanitized_json(safe_headers, key),
            'elapsed_seconds': elapsed, 'response_body_sha256': digest(data),
            'response_body_bytes': len(data), 'post_request_invocations': 1,
            'redirect_followed': False, 'billing_status': 'UNKNOWN_UNLESS_USAGE_PROVIDED'})
        require(len(data) <= 100 * 1024 * 1024, 'Response too large: STOP, budget occupied')
        if safe_headers.get('content-type', '').lower().startswith('image/') and 200 <= status < 300:
            payload = {'response_format': 'direct_image'}
            raw = data
        else:
            payload = json.loads(data.decode('utf-8'))
            write_json(run / 'response_body_redacted.json', sanitized_json(payload, key))
            require(200 <= status < 300, 'Non-success HTTP status: STOP, no retry')
            require(isinstance(payload, dict) and isinstance(payload.get('data'), list)
                    and len(payload['data']) == 1, 'Expected one output: STOP')
            item = payload['data'][0]
            if item.get('b64_json'):
                raw = base64.b64decode(item['b64_json'], validate=True)
            elif item.get('url'):
                raw = download_once(item['url'], run)
            else:
                raise ValueError('Image missing in response: STOP')
        # Save untouched bytes BEFORE Pillow decode, resize, or audit.
        with (run / 'raw_output.bin').open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            image_info = {'format': image.format, 'size_wh': list(image.size), 'mode': image.mode}
        extension = {'PNG': 'png', 'JPEG': 'jpg', 'WEBP': 'webp'}.get(image_info['format'], 'image')
        named = run / ('raw_output.' + extension)
        with named.open('xb') as f:
            f.write(raw)
        write_json(run / 'response_manifest.json', {
            'status': 'FIRST_CALL_RAW / PENDING_HUMAN_REVIEW', 'image': image_info,
            'raw_output_path': str(named), 'raw_output_sha256': digest(raw),
            'usage': sanitized_json(payload.get('usage'), key),
            'response_model': sanitized_json(payload.get('model'), key),
            'revised_prompt': sanitized_json(payload.get('data', [{}])[0].get('revised_prompt'), key)
                              if payload.get('data') else None,
            'requested_model': MODEL, 'upstream_identity': 'UNVERIFIED',
            'post_request_invocations': 1, 'elapsed_seconds': time.monotonic() - start,
            'raw_pair_geometry': 'PENDING_HUMAN_REVIEW', 'train_pair_eligible': False})
        key = None
        from audit_saved_output import audit
        audit(named, run / 'offline_audit')
        write_json(run / 'TERMINAL.json', {'status': 'FIRST_CALL_RAW / PENDING_HUMAN_REVIEW',
                   'generation_posts_maximum': 1, 'finished_utc': utc(), 'budget_stays_occupied': True})
        print('ONE_CALL_FINISHED; PENDING_HUMAN_REVIEW; STOP')
    except Exception:
        # Raw transport exceptions can contain signed paths; never persist/print exception text.
        write_json(run / 'STOP_FAILURE.json', {'status': 'STOP_NO_RETRY_BUDGET_OCCUPIED',
                   'post_request_invoked': post_invoked, 'wire_delivery_or_billing': 'UNKNOWN',
                   'elapsed_seconds': time.monotonic() - start, 'utc': utc()})
        raise RuntimeError('Single-call failed or audit incomplete; budget occupied; no retry') from None
    finally:
        key = None
        if connection is not None:
            connection.close()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--confirm', default='')
    parser.add_argument('--gate-receipt', type=Path, default=ROOT / 'gate_approval.json')
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args()
    if args.live:
        live(args.confirm, args.gate_receipt)
    else:
        require(args.output_directory is not None, 'Offline output directory required')
        dry_run(args.output_directory)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('STOP: validation/transport/audit failed; no retry; see local sanitized records.', file=sys.stderr)
        sys.exit(1)
