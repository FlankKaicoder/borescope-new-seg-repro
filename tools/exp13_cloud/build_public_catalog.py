"""Finalize text-only publication lists; no runtime experiment modules imported."""
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for path in sorted((ROOT / 'configs/exp13_cloud').glob('*.txt')):
        data = path.read_bytes()
        text = data.decode('utf-8')
        # Historical G1B/G1E transmit stripped text; G1F-1A transmits exact file text.
        exact = path.name == 'CORROSION2_ROI_REPAIR_V1.txt'
        sent = text if exact else text.strip()
        rows.append({'prompt_id': path.stem, 'public_path': path.relative_to(ROOT).as_posix(),
                     'file_sha256': hashlib.sha256(data).hexdigest(),
                     'request_string_sha256': hashlib.sha256(sent.encode('utf-8')).hexdigest(),
                     'encoding': 'UTF-8', 'has_bom': data.startswith(b'\xef\xbb\xbf'),
                     'ends_in_newline': data.endswith((b'\n', b'\r')),
                     'request_rule': 'EXACT_NO_STRIP_NO_NEWLINE' if exact else 'HISTORICAL_TEXT_STRIP'})
    target = ROOT / 'docs/exp13_cloud/prompt_hashes.csv'
    with target.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    allowed = ['.gitignore', '.gitattributes', 'docs/PROJECT_STATE.md', 'docs/DECISION_LOG.md',
               'docs/experiment_index.md', 'results/experiment_registry.csv']
    for directory in ['tools/exp13_cloud', 'configs/exp13_cloud', 'docs/exp13_cloud']:
        allowed.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / directory).rglob('*')
                       if path.is_file() and '__pycache__' not in path.parts)
    allowlist = 'docs/exp13_cloud/public_allowlist.txt'
    allowed.append(allowlist)
    allowed = sorted(set(allowed))
    if any(Path(name).suffix.lower() not in {'.md', '.txt', '.py', '.ps1', '.json', '.csv', ''}
           for name in allowed):
        raise ValueError('Unexpected public file type')
    with (ROOT / allowlist).open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(allowed) + '\n')
    print('PUBLIC_CATALOG_CREATED', len(allowed), 'explicit paths')


if __name__ == '__main__':
    main()
