"""Check new auditor compatibility using only one committed metadata epoch."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    root, record = args.root.resolve(strict=True), args.record.resolve(strict=True)
    assert record.is_relative_to(root)
    script = record / 'full_replay_audit.py'
    spec = importlib.util.spec_from_file_location('exp18_independent_metadata_auditor', script)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    inputs = auditor.Inputs()
    runtime, protocol = auditor.check_runtime(root / 'code_a10', 200, inputs)
    auditor.check_runtime(root / 'code_a10', 100, inputs)
    historical = inputs.json(runtime['historical_runtime_config'], auditor.PINNED['historical_runtime'])
    identity = inputs.json(historical['native_identity'], auditor.PINNED['native_identity'])
    dataset = Path(historical['release']) / 'dataset'
    sources = {str((dataset / s['native_image_relative']).resolve()): s for s in identity['sources']}
    commit_path = root / 'full_a10/replay/epochs/epoch_000.commit.json'
    commit = inputs.json(commit_path)
    chunk = inputs.remember(commit_path.parent / commit['chunk'], commit['chunk_sha256'])
    import scipy
    from scipy.optimize import linear_sum_assignment
    assert scipy.__version__ == '1.18.0'
    count, totals = 0, [0, 0, 0, 0]
    with gzip.open(chunk, 'rt', encoding='utf-8') as stream:
        for line in stream:
            data = json.loads(line)
            values = auditor.matching_audit(data, linear_sum_assignment)
            auditor.trace_identity_audit(data, [str(Path(t['im_file']).resolve()) for t in data['traces']], sources)
            totals = [a + b for a, b in zip(totals, values)]
            count += 1
    assert count == 21
    assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
    inputs.postcheck()
    result = {'status': 'PASS_AUDITOR_SCHEMA_COMPATIBILITY_ONLY', 'audited_batches': count,
              'full_4200_acceptance': False, 'counts_K_S_N_reuse': totals,
              'scope': 'FIRST_COMMITTED_EPOCH_METADATA_ONLY_NOT_FINAL_ACCEPTANCE',
              'auditor_sha256': auditor.sha(script), 'chunk_sha256': commit['chunk_sha256'],
              'scipy': scipy.__version__, 'real_model_constructions': 0, 'GPU_compute': 0,
              'VAL_passes': 0, 'TEST_accesses': 0}
    with (record / 'auditor_schema_smoke.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
