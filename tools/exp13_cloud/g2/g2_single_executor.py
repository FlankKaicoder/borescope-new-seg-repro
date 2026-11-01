"""One-source G2 executor. Dry-run is offline; Live is fail-closed and single-shot."""
import argparse
import base64
import ctypes
import email.policy
import hashlib
import http.client
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageChops, ImageDraw


REPO = Path(__file__).resolve().parents[3]
BUNDLE_PATH = REPO / 'configs/exp13_cloud/g2/execution_bundle.json'
PROTOCOL_PATH = REPO / 'configs/exp13_cloud/g2/pilot_protocol_v2.json'
PROMPT_PATHS = {
    'Crack': REPO / 'configs/exp13_cloud/g2/CRACK_RESTORATION_V1.txt',
    'corrosion': REPO / 'configs/exp13_cloud/g2/CORROSION_RESTORATION_V1.txt',
}
SOURCE_IDS = ('260', '470', '23')
SOURCE_TOKENS = {
    '260': 'G2_CRACK260_ONE_PAID_POST',
    '470': 'G2_CRACK470_ONE_PAID_POST',
    '23': 'G2_CORROSION23_ONE_PAID_POST',
}
BASE_URL = 'https://direct-api.ccccapi.cc'
MODEL = 'gpt-image-2.5-sunburst'
ENDPOINT_PATH = '/v1/images/edits'
MAX_RESPONSE_BYTES = 100 * 1024 * 1024


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def sanitize_response(value, credential):
    if isinstance(value, dict):
        clean = {}
        for name, item in value.items():
            if name.lower() in ('url', 'image_url', 'download_url') and isinstance(item, str):
                clean[name] = safe_url_summary(item)
            else:
                clean[name] = sanitize_response(item, credential)
        return clean
    if isinstance(value, list):
        return [sanitize_response(item, credential) for item in value]
    if isinstance(value, str):
        return value.replace(credential, '[REDACTED_CREDENTIAL]') if credential else value
    return value


def bundle_lock_bytes():
    return BUNDLE_PATH.read_bytes()


def validate_bundle():
    raw = bundle_lock_bytes()
    bundle = json.loads(raw.decode('utf-8'))
    require(bundle.get('record_type') == 'G2_FROZEN_EXECUTION_BUNDLE_NOT_AUTHORIZATION',
            'Execution bundle type mismatch')
    require(bundle.get('model') == MODEL and bundle.get('base_url') == BASE_URL
            and bundle.get('endpoint_path') == ENDPOINT_PATH, 'Frozen request endpoint/model mismatch')
    require(bundle.get('source_order') == list(SOURCE_IDS), 'Frozen source order mismatch')
    require(bundle.get('max_new_g2_posts_total') == 3 and bundle.get('max_posts_per_source') == 1,
            'Frozen generation budget mismatch')
    require(bundle.get('evaluation_denominators', {}).get('new_cloud_source_results') == ['260', '470']
            and bundle['evaluation_denominators'].get('historical_corrosion_targeted_retest_results') == ['23']
            and bundle['evaluation_denominators'].get('g2_execution_and_cost_records') == ['260', '470', '23']
            and bundle['evaluation_denominators'].get('all_three_call_outcomes_required') is True
            and bundle['evaluation_denominators'].get('new_crack_group_minimum_credible_candidates') == 1
            and bundle['evaluation_denominators'].get('corrosion23_retest_credible_for_mixed_class_gate') is True
            and bundle['evaluation_denominators'].get('no_significance_or_training_claim') is True,
            'Evaluation grouping mismatch')
    for entry in bundle['files']:
        relative = Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Unsafe frozen bundle path')
        path = REPO / relative
        require(path.is_file() and file_sha(path) == entry['sha256'], 'Frozen code/config hash mismatch')
    return bundle, digest(raw)


def source_record(bundle, source_id):
    require(source_id in SOURCE_IDS, 'Unknown source: STOP')
    record = bundle['sources'].get(source_id)
    require(isinstance(record, dict) and record.get('source_id') == source_id,
            'Source missing from frozen execution bundle')
    return record


def load_and_validate_inputs(bundle, source_id, asset_root):
    record = source_record(bundle, source_id)
    require(asset_root.is_dir() and asset_root.name == bundle['private_asset_id'],
            'Private asset root ID mismatch')
    private_manifest = read_json(asset_root / 'private_three_source_manifest.json')
    require(file_sha(asset_root / 'private_three_source_manifest.json') == bundle['private_manifest']['sha256'],
            'Private source manifest SHA256 mismatch')
    require(private_manifest.get('package_zip_sha256') == bundle['pilot4_zip_sha256']
            and private_manifest.get('frozen_split_manifest_sha256') == bundle['frozen_split_manifest_sha256'],
            'Private source manifest package/split mismatch')
    sources = [item for item in private_manifest['sources'] if item.get('source_id') == source_id]
    require(len(sources) == 1, 'Expected exactly one frozen private source record')
    frozen = sources[0]
    require(frozen.get('split') == 'train' and frozen.get('source_status') == 'matched_authoritative'
            and frozen.get('frozen_split_manifest_sha256') == bundle['frozen_split_manifest_sha256'],
            'Frozen TRAIN proof missing or mismatched')
    require(frozen.get('class') == record['class'] and frozen.get('group_id') == record['train_group_id']
            and frozen.get('image_size_wh') == record['image_size_wh']
            and frozen.get('roi_xyxy_half_open') == record['roi_xyxy_half_open'],
            'Frozen source identity/ROI mismatch')

    source_dir = asset_root / 'sources' / source_id
    image_path = source_dir / frozen['source_file_names']['image']
    annotation_path = source_dir / frozen['source_file_names']['annotation']
    binary_path = source_dir / 'polygon_mask.png'
    mask_path = source_dir / 'api_rgba_mask.png'
    expected_hashes = record['sha256']
    paths = {
        'original_image': image_path,
        'original_professional_json': annotation_path,
        'roi': image_path,
        'polygon_mask': binary_path,
        'api_rgba_mask': mask_path,
    }
    for key, path in paths.items():
        require(path.is_file() and file_sha(path) == expected_hashes[key],
                f'Frozen {key} SHA256 mismatch: STOP')
    require(frozen['input_hashes']['original_image'] == expected_hashes['original_image']
            and frozen['input_hashes']['original_annotation'] == expected_hashes['original_professional_json']
            and frozen['input_hashes']['polygon_mask'] == expected_hashes['polygon_mask']
            and frozen['input_hashes']['api_rgba_mask'] == expected_hashes['api_rgba_mask'],
            'Private manifest hash chain mismatch')

    annotation = read_json(annotation_path)
    width, height = record['image_size_wh']
    with Image.open(image_path) as original:
        original.load()
        require(original.size == (width, height), 'Original image dimensions differ from frozen record')
    require([annotation.get('imageWidth'), annotation.get('imageHeight')] == [width, height],
            'Original image/JSON dimensions mismatch')
    require(Path(annotation.get('imagePath', '')).name == record['annotation_image_basename'],
            'Original JSON image identity mismatch')
    shapes = annotation.get('shapes')
    require(isinstance(shapes, list) and len(shapes) == record['polygon_count']
            and all(shape.get('shape_type', '').lower() == 'polygon'
                    and shape.get('label') == record['class'] for shape in shapes),
            'Original professional Polygon identity mismatch')
    binary = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(binary)
    for shape, expected_points in zip(shapes, record['polygon_point_counts'], strict=True):
        points = shape.get('points')
        require(isinstance(points, list) and len(points) == expected_points and len(points) >= 3,
                'Original Polygon point-count mismatch')
        for x, y in points:
            require(0 <= x < width and 0 <= y < height, 'Original Polygon out of bounds')
        draw.polygon([tuple(point) for point in points], fill=255)
    require(0 < sum(value > 0 for value in binary.get_flattened_data()) < width * height,
            'Polygon mask empty or full-frame')
    saved_binary = Image.open(binary_path).convert('L')
    mask = Image.open(mask_path)
    require(saved_binary.size == mask.size == (width, height) and mask.mode == 'RGBA',
            'Frozen mask dimensions/mode mismatch')
    require(ImageChops.difference(binary, saved_binary).getbbox() is None,
            'Saved binary mask differs from original professional Polygons')
    alpha = mask.getchannel('A')
    require(set(alpha.get_flattened_data()) == {0, 255}, 'RGBA mask alpha is not binary')
    require(ImageChops.difference(Image.eval(binary, lambda value: 255 - value), alpha).getbbox() is None,
            'RGBA mask transparency direction/coordinates mismatch')
    transparent = alpha.histogram()[0]
    require(transparent == record['mask_target_pixels'], 'RGBA mask target-pixel count mismatch')
    prompt_path = PROMPT_PATHS[record['class']]
    prompt_bytes = prompt_path.read_bytes()
    require(digest(prompt_bytes) == record['prompt_file_sha256'], 'Frozen prompt file SHA256 mismatch')
    require(prompt_bytes.endswith(b'\n') and not prompt_bytes.endswith(b'\n\n')
            and not prompt_bytes.endswith(b'\r\n') and not prompt_bytes.startswith(b'\xef\xbb\xbf'),
            'Prompt must be UTF-8 without BOM and have one terminal LF')
    prompt_wire_bytes = prompt_bytes[:-1]
    prompt_text = prompt_wire_bytes.decode('utf-8')
    require(digest(prompt_wire_bytes) == record['prompt_transport_sha256'],
            'Prompt wire-string SHA256 mismatch')
    return {
        'record': record,
        'image_path': image_path,
        'annotation_path': annotation_path,
        'binary_mask_path': binary_path,
        'mask_path': mask_path,
        'prompt_path': prompt_path,
        'prompt_text': prompt_text,
        'prompt_wire_bytes': prompt_wire_bytes,
        'prompt_file_sha256': digest(prompt_bytes),
        'prompt_transport_sha256': digest(prompt_wire_bytes),
        'transparent_pixels': transparent,
        'annotation': annotation,
    }


def encode_multipart(source_id, model, prompt_wire_bytes, image_path, mask_path):
    boundary = f'borescope_g2_{source_id}_frozen_v2'
    parts = [
        ('model', model.encode('utf-8'), None, None),
        ('prompt', prompt_wire_bytes, None, None),
        ('n', b'1', None, None),
        ('image[]', image_path.read_bytes(), image_path.name, 'image/png' if image_path.suffix.lower() == '.png' else 'image/jpeg'),
        ('mask', mask_path.read_bytes(), 'api_rgba_mask.png', 'image/png'),
    ]
    chunks = []
    boundary_bytes = boundary.encode('ascii')
    for name, value, filename, mime in parts:
        require(boundary_bytes not in value, 'Frozen multipart boundary collision')
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
        if filename:
            header += f'; filename="{filename}"\r\nContent-Type: {mime}'
        chunks.extend([(header + '\r\n\r\n').encode('utf-8'), value, b'\r\n'])
    chunks.append(f'--{boundary}--\r\n'.encode('ascii'))
    body = b''.join(chunks)
    content_type = f'multipart/form-data; boundary={boundary}'
    message = BytesParser(policy=email.policy.default).parsebytes(
        (f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n').encode() + body)
    observed = {part.get_param('name', header='content-disposition'): part.get_payload(decode=True)
                for part in message.iter_parts()}
    expected = {name: value for name, value, _, _ in parts}
    require(observed == expected, 'Constructed multipart fields/bytes differ from frozen inputs')
    return body, content_type


def request_summary(bundle_sha, source, inputs, body, content_type):
    record = inputs['record']
    return {
        'method': 'POST',
        'endpoint': BASE_URL + ENDPOINT_PATH,
        'model': MODEL,
        'n': 1,
        'source_id': source,
        'source_role': record['source_role'],
        'multipart_fields': {
            'model': {'type': 'text', 'value': MODEL},
            'prompt': {'type': 'text', 'prompt_id': record['prompt_id'],
                       'file_sha256': inputs['prompt_file_sha256'],
                       'transmitted_string_sha256': inputs['prompt_transport_sha256'],
                       'encoding': 'UTF-8',
                       'terminal_lf_rule': 'Remove exactly the one frozen terminal LF byte; no trim or other transformation.'},
            'n': {'type': 'text', 'value': '1'},
            'image[]': {'type': 'file', 'filename': inputs['image_path'].name,
                        'sha256': record['sha256']['original_image']},
            'mask': {'type': 'file', 'filename': 'api_rgba_mask.png',
                     'sha256': record['sha256']['api_rgba_mask'],
                     'format': 'RGBA PNG', 'target_alpha': 0, 'outside_alpha': 255},
        },
        'content_type': content_type,
        'multipart_body_bytes': len(body),
        'multipart_body_sha256': digest(body),
        'redirects': False,
        'automatic_retries': 0,
        'fallbacks': 0,
        'prompt_search': 0,
        'concurrency': 0,
        'execution_bundle_sha256': bundle_sha,
        'authorization_header': 'Bearer [credential only in child process memory; value not recorded]',
    }


def offline_guard(event, args):
    if event.startswith(('socket.', 'urllib.', 'http.client.', 'subprocess.')) or event == 'os.system':
        raise RuntimeError('Offline guard blocked network or child process')


def dry_run(source_id, asset_root, execution_root, output_dir):
    sys.addaudithook(offline_guard)
    bundle, bundle_sha = validate_bundle()
    require(os.environ.get('OPENAI_BASE_URL') == BASE_URL
            and os.environ.get('OPENAI_IMAGE_MODEL') == MODEL, 'Loaded relay configuration mismatch')
    key = os.environ.pop('OPENAI_API_KEY', None)
    key_present = bool(key and key.strip())
    key = None
    require(key_present, 'DPAPI loader did not provide a child credential')
    inputs = load_and_validate_inputs(bundle, source_id, asset_root)
    live_source = execution_root / 'live' / source_id
    source_lock_dir = execution_root / 'locks' / source_id
    require(not source_lock_dir.exists() and not live_source.exists(),
            'Existing live marker/output for source: STOP')
    body, content_type = encode_multipart(source_id, MODEL, inputs['prompt_wire_bytes'],
                                          inputs['image_path'], inputs['mask_path'])
    summary = request_summary(bundle_sha, source_id, inputs, body, content_type)
    require(not output_dir.exists() and output_dir.is_relative_to(execution_root / 'dry_run' / source_id),
            'Dry-run output path invalid or already exists')
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'dry_run_manifest.json', {
        'status': 'DRY_RUN_PASS / LIVE_NOT_STARTED',
        'created_utc': utc(),
        'source_id': source_id,
        'source_role': inputs['record']['source_role'],
        'source_identity': {'class': inputs['record']['class'], 'split': 'train',
                            'train_group_id': inputs['record']['train_group_id']},
        'input_sha256': inputs['record']['sha256'],
        'roi_xyxy_half_open': inputs['record']['roi_xyxy_half_open'],
        'image_size_wh': inputs['record']['image_size_wh'],
        'polygon_count': inputs['record']['polygon_count'],
        'polygon_integrity_verified': True,
        'mask_transparent_pixels': inputs['transparent_pixels'],
        'prompt_id': inputs['record']['prompt_id'],
        'prompt_file_sha256': inputs['prompt_file_sha256'],
        'prompt_transmitted_string_sha256': inputs['prompt_transport_sha256'],
        'prompt_file_encoding': 'UTF-8 without BOM; one terminal LF',
        'prompt_wire_rule': 'Remove exactly the final LF byte; no stripping, normalization, appending or rewriting.',
        'request': summary,
        'credential_present_in_child': True,
        'credential_removed_from_child_environment': True,
        'credential_logged': False,
        'network_guard': 'ACTIVE; no network calls permitted in dry run',
        'api_post_count': 0,
        'network_request_count': 0,
        'production_started_marker_created': False,
        'generation_authorized': False,
    })
    print(f'DRY_RUN_PASS source={source_id}; API_POST=0')


def validate_receipt(receipt_path, source_id, bundle_sha):
    require(receipt_path.name == 'gate_approval.json' and receipt_path.parent.name == source_id
            and receipt_path.parent.parent.name == 'authorizations', 'Unexpected approval receipt path')
    gate = read_json(receipt_path)
    record = source_record(validate_bundle()[0], source_id)
    require(gate.get('status') == 'G2_SOURCE_LIVE_APPROVED'
            and gate.get('independent_gate_pass') is True
            and gate.get('user_approved') is True, 'Independent Gate/user approval missing')
    require(gate.get('source_id') == source_id and gate.get('source_role') == record['source_role']
            and gate.get('model') == MODEL and gate.get('max_posts') == 1,
            'Receipt scope does not match this fixed source')
    require(gate.get('execution_bundle_sha256') == bundle_sha,
            'Receipt is bound to a different frozen execution bundle')
    require(isinstance(gate.get('approval_reference'), str) and gate['approval_reference'].strip(),
            'Approval reference missing')


def acquire_windows_mutex():
    require(os.name == 'nt', 'Live mode is Windows-only')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel.ReleaseMutex.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateMutexW(None, False, 'Local\\BorescopeG2CloudSingleCall')
    require(handle, 'Could not create execution mutex')
    result = kernel.WaitForSingleObject(handle, 0)
    if result not in (0, 0x80):
        kernel.CloseHandle(handle)
        raise RuntimeError('Another G2 live executor is active; no concurrency')
    return handle


def release_windows_mutex(handle):
    if handle:
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def create_started_marker(path, payload):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def safe_url_summary(url):
    parsed = urlsplit(url)
    return {'scheme': parsed.scheme, 'host': parsed.hostname, 'path_sha256': digest(parsed.path.encode()),
            'has_query': bool(parsed.query), 'full_url_recorded': False}


def retrieve_image_url(url, output_dir):
    parsed = urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password
            and parsed.port in (None, 443), 'Output URL is not allowed')
    require(parsed.hostname.lower() not in ('localhost',) and not parsed.hostname.lower().endswith('.local'),
            'Local output host rejected')
    write_json(output_dir / 'output_download_started.json', {
        'method': 'GET', 'url': safe_url_summary(url), 'redirects': False,
        'retries': 0, 'authorization_forwarded': False})
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=120)
    try:
        path = parsed.path or '/'
        if parsed.query:
            path += '?' + parsed.query
        connection.request('GET', path, headers={'Accept': 'image/*'})
        response = connection.getresponse()
        require(response.status == 200, 'Image download failed/redirected; no retry')
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        require(len(raw) <= MAX_RESPONSE_BYTES, 'Image download exceeds size limit')
        return raw
    finally:
        connection.close()


def live(source_id, confirmation, asset_root, execution_root):
    require(confirmation == SOURCE_TOKENS[source_id], 'Source-specific single-call confirmation mismatch')
    bundle, bundle_sha = validate_bundle()
    inputs = load_and_validate_inputs(bundle, source_id, asset_root)
    require(os.environ.get('OPENAI_BASE_URL') == BASE_URL
            and os.environ.get('OPENAI_IMAGE_MODEL') == MODEL, 'Loaded relay configuration mismatch')
    key = os.environ.pop('OPENAI_API_KEY', None)
    require(bool(key and key.strip()), 'DPAPI child credential missing')
    live_root = execution_root / 'live'
    source_root = live_root / source_id
    require(not source_root.exists() and not (source_root / 'STARTED.json').exists(),
            'Source has prior live output or STARTED marker; never resend')
    lock_root = execution_root / 'locks'
    source_lock_dir = lock_root / source_id
    locked_sources = [source for source in SOURCE_IDS if (lock_root / source).exists()]
    require(len(locked_sources) < 3, 'G2 three-POST budget exhausted')
    receipt_path = execution_root / 'authorizations' / source_id / 'gate_approval.json'
    validate_receipt(receipt_path, source_id, bundle_sha)
    body, content_type = encode_multipart(source_id, MODEL, inputs['prompt_wire_bytes'],
                                          inputs['image_path'], inputs['mask_path'])
    summary = request_summary(bundle_sha, source_id, inputs, body, content_type)
    mutex = acquire_windows_mutex()
    connection = None
    output_created = False
    post_invoked = False
    start_time = time.monotonic()
    try:
        # Recheck quota and exclusivity while holding the cross-process mutex.
        locked_sources = [source for source in SOURCE_IDS if (lock_root / source).exists()]
        require(len(locked_sources) < 3 and not source_root.exists() and not source_lock_dir.exists(),
                'Budget, permanent source lock, or source output path occupied')
        source_root.mkdir(parents=True, exist_ok=False)
        output_created = True
        write_json(source_root / 'request_manifest.json', {
            'status': 'FROZEN_BEFORE_POST', 'created_utc': utc(), 'request': summary,
            'approval_receipt_sha256': file_sha(receipt_path),
            'input_hashes': inputs['record']['sha256']})
        source_lock_dir.mkdir(parents=True, exist_ok=False)
        create_started_marker(source_lock_dir / 'STARTED.json', {
            'status': 'PERMANENT_BUDGET_OCCUPIED', 'created_utc': utc(),
            'source_id': source_id, 'source_role': inputs['record']['source_role'],
            'model': MODEL, 'max_posts': 1, 'execution_bundle_sha256': bundle_sha,
            'prompt_transmitted_string_sha256': inputs['prompt_transport_sha256'],
            'rule': 'Never delete, overwrite or reuse; uncertain delivery/charge consumes this source slot.'})
        key = os.environ.pop('OPENAI_API_KEY', key)
        connection = http.client.HTTPSConnection('direct-api.ccccapi.cc', timeout=600)
        post_invoked = True
        connection.request('POST', ENDPOINT_PATH, body=body, headers={
            'Authorization': 'Bearer ' + key,
            'Content-Type': content_type,
            'Content-Length': str(len(body)),
            'Accept': 'application/json',
        })
        response = connection.getresponse()
        response_body = response.read(MAX_RESPONSE_BYTES + 1)
        elapsed = time.monotonic() - start_time
        content_type_header = response.getheader('content-type', '') or ''
        headers = {name: response.getheader(name) for name in
                   ('content-type', 'content-length', 'x-request-id', 'request-id')
                   if response.getheader(name) is not None}
        write_json(source_root / 'http_response_metadata.json', {
            'http_status': response.status, 'headers': sanitize_response(headers, key),
            'elapsed_seconds': elapsed, 'response_body_sha256': digest(response_body),
            'response_body_bytes': len(response_body), 'post_request_invocations': 1,
            'redirect_followed': False, 'billing_status': 'UNKNOWN_UNLESS_USAGE_PROVIDED'})
        require(len(response_body) <= MAX_RESPONSE_BYTES, 'Response exceeds size limit; slot consumed')
        require(200 <= response.status < 300, 'Non-success HTTP response; slot consumed; no retry')
        if content_type_header.lower().startswith('image/'):
            raw_output = response_body
            payload = {}
            response_format = 'direct_image'
        else:
            payload = json.loads(response_body.decode('utf-8'))
            require(isinstance(payload.get('data'), list) and len(payload['data']) == 1,
                    'Expected exactly one image result; slot consumed')
            item = payload['data'][0]
            if item.get('b64_json'):
                raw_output = base64.b64decode(item['b64_json'], validate=True)
                response_format = 'b64_json'
            elif item.get('url'):
                raw_output = retrieve_image_url(item['url'], source_root)
                response_format = 'url_download'
            else:
                raise RuntimeError('Image absent from response; slot consumed')
        with (source_root / 'raw_output.bin').open('xb') as handle:
            handle.write(raw_output)
            handle.flush()
            os.fsync(handle.fileno())
        with Image.open(io.BytesIO(raw_output)) as image:
            image.load()
            image_info = {'format': image.format, 'mode': image.mode, 'size_wh': list(image.size)}
        extension = {'PNG': 'png', 'JPEG': 'jpg', 'WEBP': 'webp'}.get(image_info['format'], 'image')
        with (source_root / f'raw_output.{extension}').open('xb') as handle:
            handle.write(raw_output)
            handle.flush()
            os.fsync(handle.fileno())
        usage = sanitize_response(payload.get('usage'), key) if isinstance(payload.get('usage'), dict) else None
        reported_cost = payload.get('cost')
        if reported_cost is None and usage is not None:
            reported_cost = usage.get('cost')
        reported_cost = sanitize_response(reported_cost, key)
        write_json(source_root / 'response_manifest.json', {
            'status': 'FIRST_CALL_RAW / PENDING_HUMAN_REVIEW',
            'response_format': response_format,
            'image': image_info,
            'raw_output_sha256': digest(raw_output),
            'usage': usage,
            'reported_cost': reported_cost,
            'cost_status': 'PROVIDER_REPORTED' if reported_cost is not None else 'NOT_RETURNED',
            'response_model': sanitize_response(payload.get('model'), key),
            'requested_model': MODEL,
            'upstream_identity': 'UNVERIFIED',
            'post_request_invocations': 1,
            'elapsed_seconds': time.monotonic() - start_time,
            'train_pair_eligible': False,
            'human_gates': {name: 'PENDING_HUMAN_REVIEW' for name in
                            ('DEFECT_REMOVAL', 'MECHANICAL_GEOMETRY', 'ALIGNMENT', 'PAIR_USABILITY')},
        })
        key = None
        write_json(source_root / 'TERMINAL.json', {
            'status': 'FIRST_CALL_RAW / PENDING_HUMAN_REVIEW',
            'generation_posts_maximum': 1, 'finished_utc': utc(), 'budget_stays_occupied': True})
        print(f'SINGLE_SOURCE_TERMINAL source={source_id}; POST=1; STOP')
    except Exception:
        if output_created:
            try:
                write_json(source_root / 'STOP_FAILURE.json', {
                    'status': 'STOP_NO_RETRY_BUDGET_OCCUPIED',
                    'source_id': source_id, 'post_invoked': post_invoked,
                    'delivery_or_billing': 'UNKNOWN' if post_invoked else 'NOT_SENT',
                    'elapsed_seconds': time.monotonic() - start_time, 'utc': utc()})
            except Exception:
                pass
        raise RuntimeError('Single-source execution failed/stopped; never retry; inspect private ledger') from None
    finally:
        key = None
        if connection is not None:
            connection.close()
        release_windows_mutex(mutex)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--source-id', required=True, choices=SOURCE_IDS)
    parser.add_argument('--asset-root', required=True, type=Path)
    parser.add_argument('--execution-root', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--confirm-one-paid-post', default='')
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    execution_root = args.execution_root.resolve()
    if args.dry_run:
        require(args.output_dir is not None, 'Dry-run output directory is required')
        dry_run(args.source_id, asset_root, execution_root, args.output_dir.resolve())
    else:
        live(args.source_id, args.confirm_one_paid_post, asset_root, execution_root)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('STOP: validation or execution failed; no automatic retry; inspect private sanitized records.', file=sys.stderr)
        sys.exit(1)
