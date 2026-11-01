"""Audit actual staged UTF-8 blobs; never execute/import experiment source.

Output contains only safe path names, counts, statuses and hashes. On suspected
sensitive content, print its path/category without echoing the matching value.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = '3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a'


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True).stdout


def fail(path, category):
    raise ValueError(path + ': ' + category)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def scan(path, data):
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        fail(path, 'Non-UTF8/binary candidate')
    if '\0' in text:
        fail(path, 'Binary/NUL content')
    patterns = {
        'Credential-like token': r'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})',
        'Private key/ciphertext': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|01000000d08c9ddf[0-9a-f]{64,}',
        'Signed credential URL': r'https?://[^\s"<>]+[?&](?:token|sig|key|signature)=',
    }
    for category, pattern in patterns.items():
        if re.search(pattern, text, re.I):
            fail(path, category)
    if path.endswith('.json'):
        value = json.loads(text)
        if isinstance(value, dict) and (value.get('user_approved') is True or
                                      value.get('status') == 'G1F_1_LIVE_APPROVED' or
                                      'shapes' in value or 'imageData' in value):
            fail(path, 'Approval/annotation payload is not public protocol metadata')
    if path.endswith('.py'):
        compile(text, path, 'exec')  # Syntax only: does not execute source or imports.
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.lower() in {'api_key', 'plainkey', 'secret_key'}:
                        if len(node.value.value) > 12 and not any(word in node.value.value.upper() for word in ['REDACT', 'PLACEHOLDER']):
                            fail(path, 'Literal credential assignment')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='New local report OUTSIDE this repository')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        raise ValueError('Audit output must be a fresh path outside the public checkout')
    branch = git('branch', '--show-current').decode().strip()
    if branch != 'phase/exp13-cloud-repair':
        raise ValueError('Wrong publication branch')
    if git('rev-parse', 'HEAD').decode().strip() != BASE:
        raise ValueError('Precommit HEAD must be exact remote Exp13 base')
    allowed_path = 'docs/exp13_cloud/public_allowlist.txt'
    allowed = set((ROOT / allowed_path).read_text(encoding='utf-8').splitlines())
    paths = git('diff', '--cached', '--name-only', '-z').decode().split('\0')[:-1]
    if set(paths) != allowed:
        raise ValueError('Staged paths differ from exact reviewed allowlist')
    statuses = git('diff', '--cached', '--name-status').decode()
    stats = git('diff', '--cached', '--stat').decode()
    diff = git('diff', '--cached', '--no-ext-diff', '--no-color')
    scan('ACTUAL_STAGED_DIFF', diff)
    git('diff', '--cached', '--check')
    blobs = {}
    for path in paths:
        if Path(path).name.lower() in {'started.json', 'gate_approval.json'} or re.search(r'\.(?:png|jpg|jpeg|webp|zip|dpapi|key|pem|bin)$', path, re.I):
            fail(path, 'Prohibited runtime/media/credential path')
        data = git('show', ':' + path)
        scan(path, data)
        mode = git('ls-files', '--stage', '--', path).decode().split()[0]
        if mode not in {'100644', '100755'}:
            fail(path, 'Symlink or unexpected Git file mode')
        # Worktree and staged bytes must match exactly for new cloud snapshots.
        if path.startswith(('tools/exp13_cloud/', 'configs/exp13_cloud/', 'docs/exp13_cloud/')):
            if data != (ROOT / path).read_bytes():
                fail(path, 'Staged bytes differ from reviewed worktree bytes')
        blobs[path] = data
    provenance = list(csv.DictReader(io.StringIO(blobs['docs/exp13_cloud/code_provenance.csv'].decode())))
    for row in provenance:
        actual = sha(blobs[row['public_path']])
        if actual != row['public_sha256']:
            fail(row['public_path'], 'Public snapshot SHA mismatch')
        if row['transform'].startswith('BYTE_IDENTICAL') and actual != row['source_sha256']:
            fail(row['public_path'], 'Source snapshot SHA mismatch')
    # Preserve all old registry rows and project-state sections exactly as text prefixes.
    for path in ['docs/PROJECT_STATE.md', 'docs/experiment_index.md', 'results/experiment_registry.csv']:
        old = git('show', 'HEAD:' + path).decode('utf-8-sig').replace('\r\n', '\n').rstrip('\n')
        new = blobs[path].decode('utf-8-sig').replace('\r\n', '\n')
        if not new.startswith(old + '\n'):
            fail(path, 'Historical content changed rather than appended')
    old_decisions = git('show', 'HEAD:docs/DECISION_LOG.md').decode('utf-8-sig').replace('\r\n', '\n')
    new_decisions = blobs['docs/DECISION_LOG.md'].decode('utf-8-sig').replace('\r\n', '\n')
    # New decision is inserted after the heading; all prior decisions must remain exact.
    if not new_decisions.endswith(old_decisions.split('\n', 1)[1]):
        fail('docs/DECISION_LOG.md', 'Old decision log content changed')
    registry = list(csv.DictReader(io.StringIO(blobs['docs/exp13_cloud/experiment_registry.csv'].decode())))
    if len(registry) != 8 or sum(int(row['historical_generation_posts']) for row in registry) != 3:
        raise ValueError('Historical stage/POST ledger mismatch')
    protocol = json.loads(blobs['configs/exp13_cloud/protocol_summary.json'])
    if protocol['corrosion2']['live_state'] != 'LIVE_NOT_STARTED' or protocol['corrosion2']['live_authorized']:
        raise ValueError('Preflight incorrectly promoted to live success/authority')
    media = git('diff', '--cached', '--numstat').decode()
    if any(line.startswith('-\t-\t') for line in media.splitlines()):
        raise ValueError('Staged binary diff found')
    report = {'status': 'STAGED_PUBLICATION_AUDIT_PASS', 'branch': branch, 'base': BASE,
              'staged_paths': paths, 'staged_file_count': len(paths), 'staged_name_status': statuses,
              'staged_stat': stats, 'staged_full_diff_sha256': sha(diff),
              'public_blob_sha256': {path: sha(data) for path, data in blobs.items()},
              'source_provenance_rows_verified': len(provenance), 'new_private_binaries': 0,
              'credential_pattern_hits': 0, 'approval_records': 0, 'private_maps': 0,
              'historical_document_and_registry_content_preserved': True,
              'inherited_tracked_images': 'Explicitly accepted; unchanged and not rescanned or republished as new assets',
              'historical_generation_posts': 3, 'sync_api_posts': 0,
              'experiment_code_executed': False, 'dpapi_loaded': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ['status', 'staged_file_count', 'source_provenance_rows_verified',
                                           'new_private_binaries', 'credential_pattern_hits', 'sync_api_posts']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('STOP_PUBLICATION_AUDIT: ' + str(error), file=sys.stderr)
        sys.exit(1)
