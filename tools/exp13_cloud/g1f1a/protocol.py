"""Frozen corrosion2 protocol; local validation and exact multipart encoding."""
import csv
import hashlib
import json
import os
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parent
PREVIOUS = Path(r'E:\borescope_review_assets\cloud_pilot_g1f_roi_prepare\g1f_0b_4x3')
SOURCE = PREVIOUS.parent / 'source_snapshot'
BASE = 'https://direct-api.ccccapi.cc'
MODEL = 'gpt-image-2.5-sunburst'
ENDPOINT = BASE + '/v1/images/edits'
CONFIRM = 'AUTHORIZE_G1F1_CORROSION2_ONE_PAID_POST'
EXPECTED_ROI = '887a206bb4c550f9b8ba006684a93d1089619f06bec57c3d1e6e16648dabd74a'
EXPECTED_MASK = 'da23f8bb1533a83f107bf8b2ef5d5d3f072bc0a9f5ba827393b60af107845dfc'
EXPECTED_SOURCE = '514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03'
EXPECTED_JSON = 'e4fa956d8ae736e4e851dde5fcf66e16faf2bf87deb284c81039da3e5621717a'
PROMPT = 'Remove the visible corrosion and dark damaged surface texture only within the transparent mask region. Reconstruct a plausible clean metal surface consistent with the immediately surrounding material. Preserve the original physical boundary between the metal component and the dark open cavity. Do not fill the cavity, invent metal parts, bridge mechanical gaps, or change the blade geometry. Keep all content outside the mask as unchanged as possible, including the inspection interface, logo, indicators, lighting, and surrounding structures. Do not add text, annotations, or new objects. Return one edited image.'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def validate_inputs():
    image_path = ROOT / 'inputs/roi.png'
    mask_path = ROOT / 'inputs/mask_rgba.png'
    prompt_path = ROOT / 'prompt_v1.txt'
    require(file_sha(image_path) == EXPECTED_ROI, 'Frozen ROI SHA mismatch: STOP')
    require(file_sha(mask_path) == EXPECTED_MASK, 'Frozen mask SHA mismatch: STOP')
    require(file_sha(PREVIOUS / 'roi_b_4x3_original.png') == EXPECTED_ROI, 'Historical ROI changed: STOP')
    require(file_sha(PREVIOUS / 'roi_b_4x3_api_rgba_mask.png') == EXPECTED_MASK, 'Historical mask changed: STOP')
    require(file_sha(SOURCE / 'original_image.png') == EXPECTED_SOURCE, 'Source SHA mismatch: STOP')
    require(file_sha(SOURCE / 'original_annotation.json') == EXPECTED_JSON, 'JSON SHA mismatch: STOP')
    prompt_bytes = prompt_path.read_bytes()
    require(prompt_bytes == PROMPT.encode('utf-8'), 'Frozen prompt text/encoding/newline mismatch: STOP')
    ann = json.loads((SOURCE / 'original_annotation.json').read_text(encoding='utf-8'))
    require(ann['imagePath'] == '2.png' and (ann['imageWidth'], ann['imageHeight']) == (640, 480), 'Wrong source')
    require(len(ann['shapes']) == 1 and ann['shapes'][0]['label'] == 'corrosion'
            and ann['shapes'][0]['shape_type'] == 'polygon', 'Wrong polygon identity')
    points = [tuple(point) for point in ann['shapes'][0]['points']]
    require(len(points) == 26 and all(0 <= x < 320 and 0 <= y < 240 for x, y in points), 'Clipped polygon')
    with (SOURCE / 'source_mapping.csv').open(encoding='utf-8-sig', newline='') as f:
        row = next(r for r in csv.DictReader(f) if r['source_key'] == 'corrosion2')
    require(row['split'] == 'train' and row['group_id'] == 'singleton_2'
            and row['image_sha256'] == EXPECTED_SOURCE and row['json_sha256'] == EXPECTED_JSON, 'Not frozen TRAIN')
    roi = Image.open(image_path)
    mask = Image.open(mask_path)
    require(roi.size == mask.size == (320, 240) and mask.mode == 'RGBA'
            and roi.format == mask.format == 'PNG', 'Image/mask mode or dimensions mismatch')
    source = Image.open(SOURCE / 'original_image.png').convert('RGB')
    require(ImageChops.difference(source.crop((0, 0, 320, 240)), roi.convert('RGB')).getbbox() is None, 'ROI pixels differ')
    binary = Image.new('L', (640, 480), 0)
    ImageDraw.Draw(binary).polygon(points, fill=255)
    target = binary.crop((0, 0, 320, 240))
    alpha = mask.getchannel('A')
    require(set(alpha.tobytes()) == {0, 255}, 'Nonbinary mask alpha')
    require(alpha.histogram()[0] == binary.histogram()[255] == 2394, 'Mask pixel count mismatch')
    require(ImageChops.difference(target, Image.eval(alpha, lambda v: 255-v)).getbbox() is None, 'Wrong alpha/polygon')
    require(target.getbbox() == (40, 64, 160, 113), 'Polygon bbox mismatch')
    return {'status': 'FROZEN_INPUT_PASS', 'source': 'corrosion2', 'split': 'train',
            'group_id': 'singleton_2', 'roi_xyxy_half_open': [0, 0, 320, 240], 'size_wh': [320, 240],
            'roi_sha256': EXPECTED_ROI, 'mask_sha256': EXPECTED_MASK,
            'polygon_complete': True, 'mask_transparent_pixels': 2394, 'mask_coverage_percent': 3.1171875,
            'inside_alpha': 0, 'outside_alpha': 255,
            'prompt_text': PROMPT, 'prompt_file_sha256': digest(prompt_bytes),
            'prompt_request_string_sha256': digest(PROMPT.encode('utf-8')),
            'encoding': 'UTF-8, no BOM', 'trailing_newline': 'NONE; no strip/rstrip/normalization',
            'request_prompt_rule': 'Decode exact file bytes as UTF-8; require equality with frozen literal.'}


def build_multipart(boundary='borescope_g1f1_corrosion2_frozen_v1'):
    # Fixed boundary is safe only if absent from all payloads; explicitly check it.
    fields = [('model', MODEL.encode(), None),
              ('prompt', (ROOT / 'prompt_v1.txt').read_bytes(), None), ('n', b'1', None),
              ('image[]', (ROOT / 'inputs/roi.png').read_bytes(), 'roi.png'),
              ('mask', (ROOT / 'inputs/mask_rgba.png').read_bytes(), 'mask_rgba.png')]
    chunks = []
    for name, value, filename in fields:
        require(boundary.encode() not in value, 'Multipart boundary collision')
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
        if filename:
            header += f'; filename="{filename}"\r\nContent-Type: image/png'
        chunks.extend([(header + '\r\n\r\n').encode('utf-8'), value, b'\r\n'])
    chunks.append(f'--{boundary}--\r\n'.encode())
    return b''.join(chunks), 'multipart/form-data; boundary=' + boundary


def request_metadata():
    checks = validate_inputs()
    body, content_type = build_multipart()
    return {'status': 'PREPARED_NOT_SENT', 'endpoint': ENDPOINT, 'method': 'POST', 'model': MODEL,
            'n': 1, 'multipart_fields': ['model', 'prompt', 'n', 'image[]', 'mask'],
            'image_field_note': 'image[] is the independent image-file field inherited from verified G1E; mask is a separate file field.',
            'fields': {'image[]': {'file': str(ROOT / 'inputs/roi.png'), 'sha256': EXPECTED_ROI},
                       'mask': {'file': str(ROOT / 'inputs/mask_rgba.png'), 'sha256': EXPECTED_MASK},
                       'prompt': {'file': str(ROOT / 'prompt_v1.txt'), 'sha256': checks['prompt_request_string_sha256']}},
            'size': 'OMITTED', 'quality': 'OMITTED', 'response_format': 'OMITTED',
            'content_type': content_type, 'body_bytes': len(body), 'body_sha256': digest(body),
            'authorization_header': 'Bearer [SECRET PROVIDED ONLY IN CHILD MEMORY]',
            'redirects': False, 'automatic_retries': 0, 'fallbacks': 0, 'prompt_search': 0,
            'concurrency': 0, 'max_generation_posts': 1, 'timeout_seconds': 600}


def validate_bundle():
    lock = json.loads((ROOT / 'bundle_lock.json').read_text(encoding='utf-8'))
    for relative, expected in lock['files'].items():
        require(file_sha(ROOT / relative) == expected, 'Frozen bundle SHA mismatch: STOP')
    return lock


def create_started_marker(path, payload):
    # x => O_EXCL/CreateNew. Never remove, truncate, overwrite, or reuse this marker.
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def validate_approval(receipt, confirmation):
    require(confirmation == CONFIRM, 'Dedicated single-paid-request confirmation required')
    require(receipt == ROOT / 'gate_approval.json', 'Use the fixed independent Gate receipt path')
    gate = json.loads(receipt.read_text(encoding='utf-8'))
    require(gate.get('status') == 'G1F_1_LIVE_APPROVED'
            and gate.get('independent_gate_pass') is True and gate.get('user_approved') is True,
            'Independent Gate and user approval required')
    require(gate.get('source') == 'corrosion2' and gate.get('model') == MODEL
            and gate.get('max_posts') == 1, 'Wrong approval scope')
    require(gate.get('bundle_lock_sha256') == file_sha(ROOT / 'bundle_lock.json'), 'Approval binds a different bundle')
    require(bool(gate.get('approval_reference')), 'Missing approval provenance')
    return gate
