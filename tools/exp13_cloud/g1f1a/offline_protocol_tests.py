"""No live CLI/function execution, HTTP mocks, credentials, or network requests."""
import ast
import email.policy
import json
import sys
from email.parser import BytesParser
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import (ROOT, PROMPT, EXPECTED_ROI, EXPECTED_MASK, digest, build_multipart,
                      create_started_marker, validate_approval, write_json, validate_inputs)
from audit_saved_output import observe, delta_stats, audit


def guard(event, args):
    if event.startswith(('socket.', 'urllib.', 'http.client.', 'subprocess.')):
        raise RuntimeError('Offline tests forbid network/children')


sys.addaudithook(guard)


def main():
    target = ROOT / 'offline_tests'
    target.mkdir(exist_ok=False)
    validate_inputs()
    body, content_type = build_multipart()
    message = BytesParser(policy=email.policy.default).parsebytes(
        ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + body)
    parts = {part.get_param('name', header='content-disposition'): part.get_payload(decode=True)
             for part in message.iter_parts()}
    assert set(parts) == {'image[]', 'mask', 'prompt', 'model', 'n'}
    assert parts['prompt'] == PROMPT.encode('utf-8')
    assert digest(parts['image[]']) == EXPECTED_ROI and digest(parts['mask']) == EXPECTED_MASK
    assert parts['n'] == b'1' and parts['model'] == b'gpt-image-2.5-sunburst'
    marker = target / 'TEST_ONLY_STARTED.json'
    create_started_marker(marker, {'TEST_ONLY': True, 'post_count': 0})
    original = marker.read_bytes()
    try:
        create_started_marker(marker, {'SHOULD_NOT_REPLACE': True})
    except FileExistsError:
        pass
    else:
        raise AssertionError('Marker allowed second acquisition')
    assert marker.read_bytes() == original
    partial = target / 'TEST_ONLY_PARTIAL_MARKER.json'
    with partial.open('xb') as f:
        f.write(b'{')
    try:
        create_started_marker(partial, {'SHOULD_NOT_REPLACE': True})
    except FileExistsError:
        pass
    else:
        raise AssertionError('Partial marker allowed second acquisition')
    hold = json.loads((ROOT / 'gate_approval.TEMPLATE.json').read_text())
    assert hold['independent_gate_pass'] is False and hold['user_approved'] is False
    try:
        validate_approval(ROOT / 'gate_approval.json', '')
    except (ValueError, FileNotFoundError):
        pass
    else:
        raise AssertionError('Missing gate/confirmation accepted')
    same, valid, transform = observe(Image.new('RGB', (640, 480), (55, 55, 55)))
    assert transform['same_aspect_ratio'] and transform['letterbox_padding_pixels'] == 0
    different, valid_diff, diff_transform = observe(Image.new('RGB', (400, 400), (55, 55, 55)))
    assert not diff_transform['same_aspect_ratio'] and diff_transform['rounded_resize_wh'] == [240, 240]
    assert diff_transform['offset_xy'] == [40, 0] and diff_transform['letterbox_padding_pixels'] == 19200
    mask = Image.new('L', (320, 240), 0)
    _, stats = delta_stats(Image.new('RGB', (320, 240), (55, 55, 55)), different, mask, valid_diff)
    assert stats['outside_mask']['mean_absolute_rgb_difference'] == 0
    assert stats['outside_mask']['valid_pixels'] == 57600
    # Clearly labelled artificial fixture exercises offline nine-view rendering;
    # it is not an API result and is never put into live_result.
    fixture = target / 'TEST_ONLY_SYNTHETIC_IMAGE.png'
    Image.new('RGB', (400, 400), (55, 55, 55)).save(fixture)
    audit(fixture, target / 'TEST_ONLY_AUDIT_RENDER')
    rendered = json.loads((target / 'TEST_ONLY_AUDIT_RENDER/audit_report.json').read_text(encoding='utf-8'))
    assert rendered['train_pair_eligible'] is False
    assert all(v == 'PENDING_HUMAN_REVIEW' for v in rendered['gates'].values())
    source = (ROOT / 'single_repair.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    live_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'live')
    marker_line = next(n.lineno for n in ast.walk(live_fn) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == 'create_started_marker')
    post_line = next(n.lineno for n in ast.walk(live_fn) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'request'
                     and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'POST')
    assert marker_line < post_line
    assert not (ROOT / 'STARTED.json').exists() and not (ROOT / 'live_result').exists()
    write_json(target / 'test_report.json', {
        'status': 'PASS', 'api_post_count': 0, 'network_requests': 0,
        'live_cli_or_function_called': False,
        'multipart_tests': ['exact independent image/mask fields', 'exact prompt bytes after MIME parsing',
                            'PNG SHA equality', 'model and n fixed'],
        'marker_tests': ['exclusive creation', 'existing marker cannot be overwritten',
                         'partial marker consumes budget', 'marker precedes POST by AST order'],
        'gate_tests': ['HOLD template not approved', 'missing receipt/confirmation rejected'],
        'geometry_tests': ['same-aspect uniform resize', 'mismatched aspect letterbox without stretching',
                           'diagnostic stats exclude padding'],
        'production_marker_created': False})
    print('OFFLINE_PROTOCOL_TESTS_PASS; LIVE_NOT_INVOKED; API_POST=0')


if __name__ == '__main__':
    main()
