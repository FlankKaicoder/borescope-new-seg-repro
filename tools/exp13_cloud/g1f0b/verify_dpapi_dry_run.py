"""Fixed offline child. No live option, SDK, HTTP client, or request execution."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

NETWORK_EVENTS = []


def reject_network_and_children(event, args):
    if event.startswith(('socket.', 'urllib.', 'http.client.', 'subprocess.')) or event in ('os.system', 'os.posix_spawn'):
        NETWORK_EVENTS.append(event)
        raise RuntimeError('Offline-only operation blocked')


sys.addaudithook(reject_network_and_children)

from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value):
    if not value:
        raise ValueError('Offline validation failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    output = Path(args.output_directory).resolve()
    require(output.parent == ROOT / 'dry_runs' and not output.exists())
    # Presence only: never return, print, hash, persist, or format the key.
    secret = os.environ.pop('OPENAI_API_KEY', None)
    key_present = bool(secret and secret.strip())
    secret = None
    base = os.environ.get('OPENAI_BASE_URL', '')
    model = os.environ.get('OPENAI_IMAGE_MODEL', '')
    require(key_present)
    require(base == 'https://direct-api.ccccapi.cc')
    require(model == 'gpt-image-2.5-sunburst')
    m = json.loads((ROOT / 'roi_b_4x3_manifest.json').read_text(encoding='utf-8'))
    source_path = Path(m['source_image']['path'])
    ann_path = Path(m['source_json']['path'])
    require(sha(source_path) == m['source_image']['sha256'])
    require(sha(ann_path) == m['source_json']['sha256'])
    ann = json.loads(ann_path.read_text(encoding='utf-8'))
    require(ann['imagePath'] == '2.png' and (ann['imageWidth'], ann['imageHeight']) == (640, 480))
    require(len(ann['shapes']) == 1 and ann['shapes'][0]['label'] == 'corrosion')
    points = [tuple(point) for point in ann['shapes'][0]['points']]
    require(all(0 <= x < 320 and 0 <= y < 240 for x, y in points))
    source = Image.open(source_path).convert('RGB')
    roi = Image.open(ROOT / 'roi_b_4x3_original.png')
    mask = Image.open(ROOT / 'roi_b_4x3_binary_mask.png')
    api = Image.open(ROOT / 'roi_b_4x3_api_rgba_mask.png')
    require(roi.size == mask.size == api.size == (320, 240))
    require(api.format == 'PNG' and api.mode == 'RGBA' and mask.mode == 'L')
    require(ImageChops.difference(source.crop((0, 0, 320, 240)), roi.convert('RGB')).getbbox() is None)
    # Independently rasterize at full source resolution and compare saved mask.
    full_mask = Image.new('L', source.size, 0)
    ImageDraw.Draw(full_mask).polygon(points, fill=255)
    require(ImageChops.difference(full_mask.crop((0, 0, 320, 240)), mask).getbbox() is None)
    target_pixels = mask.histogram()[255]
    require(target_pixels == full_mask.histogram()[255] == 2394)
    require(sum(mask.histogram()[1:255]) == 0)
    require(ImageChops.difference(mask, Image.eval(api.getchannel('A'), lambda value: 255-value)).getbbox() is None)
    require(set(api.getchannel('A').tobytes()) == {0, 255})
    for info in m['files'].values():
        require(sha(info['path']) == info['sha256'])
    prompt_path = ROOT / 'prompt_CLOUD_RESTORATION_V1.txt'
    require(prompt_path.read_bytes() == (ROOT.parent / 'configs/prompt_CLOUD_RESTORATION_V1.txt').read_bytes())
    request = {
        'status': 'POST_NOT_SENT', 'source': 'corrosion2', 'roi': 'ROI_B_4x3',
        'endpoint': base + '/v1/images/edits', 'method_if_later_authorized': 'POST',
        'model': model, 'n': 1, 'size': 'OMITTED', 'quality': 'OMITTED',
        'multipart_fields': [
            {'name': 'model', 'type': 'text', 'value': model},
            {'name': 'prompt', 'type': 'text', 'prompt_id': 'CLOUD_RESTORATION_V1',
             'path': str(prompt_path), 'sha256': sha(prompt_path)},
            {'name': 'n', 'type': 'text', 'value': '1'},
            {'name': 'image[]', 'type': 'file', 'content_type': 'image/png',
             'path': str(ROOT / 'roi_b_4x3_original.png'), 'sha256': sha(ROOT / 'roi_b_4x3_original.png')},
            {'name': 'mask', 'type': 'file', 'content_type': 'image/png',
             'path': str(ROOT / 'roi_b_4x3_api_rgba_mask.png'), 'sha256': sha(ROOT / 'roi_b_4x3_api_rgba_mask.png')}
        ],
        'authentication': 'Child credential presence verified; secret intentionally omitted.',
        'automatic_retries': 0, 'fallbacks': 0, 'prompt_search': 0,
        'api_post_count': 0, 'live_authorized': False,
    }
    report = {
        'status': 'DRY_RUN_PASS', 'base_url': base, 'model': model,
        'credential_present_in_child': key_present,
        'credential_removed_from_child_environment': 'OPENAI_API_KEY' not in os.environ,
        'credential_recorded': False,
        'roi_size_wh': list(roi.size), 'source_and_json_sha_verified': True,
        'pixel_exact_source_crop': True, 'full_polygon_retained': True,
        'mask_pixels': target_pixels, 'mask_coverage_percent': target_pixels / 76800 * 100,
        'alpha_semantics_verified': True, 'fixed_prompt_byte_identical': True,
        'network_audit_hook_installed': True, 'blocked_network_or_child_events': len(NETWORK_EVENTS),
        'api_post_count': 0, 'network_requests': 0, 'automatic_retries': 0,
        'fallbacks': 0, 'prompt_search': 0, 'gpu': 0, 'training': 0, 'test_access': 0,
        'request_metadata': request
    }
    require(not NETWORK_EVENTS)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'dry_run_report.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print('DRY_RUN_PASS; API_POST=0; CREDENTIAL_NOT_LOGGED')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Do not propagate third-party exception text through credential-bearing processes.
        print('DRY_RUN_FAILED; NO_REQUEST_SENT', file=sys.stderr)
        sys.exit(1)
