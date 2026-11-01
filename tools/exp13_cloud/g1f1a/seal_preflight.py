"""Read-only verification followed by additive final SHA inventory; no credentials."""
import json
from pathlib import Path
from protocol import ROOT, file_sha, validate_bundle, validate_inputs, write_json, require
from prepare_preflight import history_snapshot

validate_bundle()
validate_inputs()
summary = json.loads((ROOT / 'preflight_summary.json').read_text(encoding='utf-8'))
require(summary['api_post_count'] == 0 and not summary['live_executed'], 'Unexpected live state')
for name in ['STARTED.json', 'live_result', 'gate_approval.json']:
    require(not (ROOT / name).exists(), 'Live state or approval unexpectedly exists')
before = json.loads((ROOT / 'history_before_sha256.json').read_text(encoding='utf-8'))
require(before == history_snapshot(), 'Historical assets changed')
loaders = list(ROOT.rglob('loader_result_*.json'))
for loader in loaders:
    data = json.loads(loader.read_text(encoding='utf-8-sig'))
    require(data['api_post_count'] == 0 and not data['live_mode'], 'Unexpected live loader')
for path in ROOT.rglob('*.json'):
    # TEST_ONLY_PARTIAL_MARKER intentionally demonstrates malformed-lock refusal.
    if path.name != 'TEST_ONLY_PARTIAL_MARKER.json':
        json.loads(path.read_text(encoding='utf-8-sig'))
write_json(ROOT / 'final_sha256_inventory.json', {
    'status': summary['status'], 'api_post_count_total': 0, 'live_executed': False,
    'dpapi_dry_runs_including_preserved_preliminary_revision': len(loaders),
    'historical_files_unchanged': len(before),
    'production_marker_exists': False, 'approved_receipt_exists': False,
    'files': {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(ROOT.rglob('*'))
              if p.is_file() and 'final_sha256_inventory.json' != p.name}})
print('FINAL_BUNDLE_AND_HISTORY_PASS; API_POST_TOTAL=0; LIVE_NOT_STARTED')
