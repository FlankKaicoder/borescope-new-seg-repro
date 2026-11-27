"""Recheck frozen EXP18 CPU inputs and source bytes after full acceptance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

BASELINE_SHA = '55f35b49d66a9641680d33570349a07cda70f0afa11e474c4b8ebc947f9f52a2'
MANIFEST_SHA = 'd545d9e59232aa60c5109fac7128ba43dfbb0207cd85f9614ec44e823d48408e'
SOURCE_SHA = 'db2a7f79ce9bd1f92ca98251cd427de4822d10d15e91306d5adb88a3f7708b58'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    root, record = args.root.resolve(strict=True), args.record.resolve(strict=True)
    require(record.is_relative_to(root), 'Record escaped EXP18 root')
    code = root / 'code_a10'
    baseline_path = root / 'closure/readonly_input_and_source_postcheck.json'
    manifest_path = record / 'source_manifest.json'
    require(sha(baseline_path) == BASELINE_SHA, 'Historical acceptance receipt changed')
    require(sha(manifest_path) == MANIFEST_SHA, 'Frozen source manifest changed')
    require(sha(root / 'source_a10.tar') == SOURCE_SHA, 'Frozen source package changed')
    baseline = json.loads(baseline_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    preflight = json.loads((record / 'preflight.json').read_text())
    launch = json.loads((record / 'launch.json').read_text())
    require(preflight['baseline_receipt_sha256'] == BASELINE_SHA
            and preflight['source_manifest_sha256'] == MANIFEST_SHA
            and sha(record / 'preflight.json') == launch['preflight_sha256'],
            'Preflight chain changed')
    require(sha(record / 'resume_control.py') == launch['control_sha256'], 'Launch controller changed')
    require(len(baseline['inputs']) == len(preflight['checked_inputs']) == 1899
            and len(manifest['files']) == 446, 'Input/source population changed')
    before = {r['path']: r for r in preflight['checked_inputs']}
    checked = []
    for row in baseline['inputs']:
        path = Path(row['path']).resolve(strict=True)
        require('test' not in {p.casefold() for p in path.parts}, 'TEST file forbidden')
        wanted = before[row['path']]
        stat = path.stat()
        require(sha(path) == row['sha256'] == wanted['sha256']
                and stat.st_size == wanted['bytes'] and stat.st_mtime_ns == wanted['mtime_ns'],
                'Read-only input changed: ' + str(path))
        checked.append({'path': str(path), 'bytes': stat.st_size,
                        'mtime_ns': stat.st_mtime_ns, 'sha256': row['sha256']})
    for row in manifest['files']:
        path = (code / row['path']).resolve(strict=True)
        require(path.is_relative_to(code) and sha(path) == row['sha256'], 'Frozen source changed: ' + row['path'])
    sys.path.insert(0, str(code))
    from tools.exp18.replay import replay_source_manifest
    require(replay_source_manifest() == baseline['runtime_source'], 'Native environment source changed')
    torch_dist = importlib.metadata.distribution('torch')
    for name, expected in baseline['additional_Torch_sources'].items():
        require(sha(torch_dist.locate_file('torch/' + name)) == expected, 'Torch source changed: ' + name)
    require('torch' not in sys.modules and 'ultralytics' not in sys.modules, 'Postcheck imported model runtime')
    replay_exit = json.loads((record / 'exit.json').read_text())
    coverage = json.loads((record / 'coverage/summary.json').read_text())
    audit = json.loads((record / 'full_audit/summary.json').read_text())
    require(replay_exit['returncode'] == 0, 'Replay process did not exit successfully')
    require(coverage['audited_epochs'] == 200 and coverage['audited_batches'] == 4200
            and coverage['maximum_assignment_recomputed'] is True
            and coverage['structural_gate'] == 'PASS', 'Coverage acceptance incomplete')
    require(audit['status'] == 'PASS_FULL_200_EPOCH_4200_BATCH_METADATA'
            and audit['completed_epochs'] == 200 and audit['completed_batches'] == 4200
            and audit['bindings_sha256'] == coverage['bindings_sha256'], 'Independent acceptance incomplete')
    reused = [
        ('tests_a10/unittest.log', '2de3e05ccb0c281f9d7f90075dfb8fb6047080d39a9f454c7a17b9a1ecaffead'),
        ('tests_a10/receipt_1795616915507367955.json', '98d4c3d613cfabe81ba8c2ac0d37e05a7ac95fd9acac162692ca11c8b55da994'),
        ('fixed_a10/receipt.json', 'a50674a42907a86ff4e34c4053494977297c7c3006078818134b85e6d851fb46')]
    for name, expected in reused:
        require(sha(root / name) == expected, 'Previous CPU evidence changed: ' + name)
    value = {'schema': 'EXP18_CPU_POSTFLIGHT_V1', 'status': 'PASS',
             'checked_inputs': checked, 'input_files': 1899, 'source_files': 446,
             'baseline_receipt_sha256': BASELINE_SHA, 'source_manifest_sha256': MANIFEST_SHA,
             'source_archive_sha256': SOURCE_SHA, 'input_size_mtime_sha_unchanged_since_preflight': True,
             'native_environment_source_matches_baseline': True, 'torch_imported': False,
             'completed_epochs': 200, 'completed_batches': 4200,
             'coverage_summary_sha256': sha(record / 'coverage/summary.json'),
             'full_audit_summary_sha256': sha(record / 'full_audit/summary.json'),
             'reused_CPU83_and_fixed4_receipts': [{'file': n, 'sha256': s} for n, s in reused],
             'CPU83_tests_rerun_in_this_session': False,
             'budgets': audit['budgets'], 'lambda_state': None, 'formal_training_authorized': False,
             'scientific_coverage_gate': 'PENDING_REVIEW', 'checked_at_unix': time.time(),
             'postflight_source_sha256': sha(__file__)}
    with (record / 'postflight.json').open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'status': 'PASS', 'input_files': 1899, 'source_files': 446,
                      'completed_batches': 4200}))


if __name__ == '__main__':
    main()
