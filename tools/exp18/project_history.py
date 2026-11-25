"""Project immutable TRAIN-only R5 identities for a smaller coverage audit."""
import argparse
import gzip
import json
import os
from pathlib import Path

from .common import require, sha256, write_json_exclusive
from .coverage_report import _old_abn

EXPECTED = 'ca1f5b6942db8d8e30fb9c1539f1aeeefc495e9f9ebde21ca9c43b2929afb58e'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    require(sha256(args.input) == EXPECTED, 'Original R5 step log SHA changed')
    count, exposure = 0, 0
    destination = args.output / 'r5_abn.jsonl.gz'
    with args.input.open(encoding='utf-8') as source, destination.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=1) as out:
            for line in source:
                record = json.loads(line)
                count += 1
                require(record['step'] == count, 'Historical step order changed')
                old = _old_abn(record)
                require(old is not None, 'Historical ABN identity fields are unavailable')
                rows = [identity for identity, weight in old.values()]
                exposure += len(rows)
                out.write((json.dumps({'step': count, 'abn_rows': rows}, ensure_ascii=False,
                                      separators=(',', ':')) + '\n').encode())
        raw.flush(); os.fsync(raw.fileno())
    require(count == 2100 and exposure == 49762, 'Historical ABN denominator changed')
    write_json_exclusive(args.output / 'receipt.json', {'source_sha256': EXPECTED, 'steps': count,
        'ABN_exposures': exposure, 'projection_sha256': sha256(destination),
        'identity_projection_only': True, 'images_or_weights_read': 0})
    print(json.dumps({'steps': count, 'exposures': exposure, 'output': str(destination)}))


if __name__ == '__main__': main()
