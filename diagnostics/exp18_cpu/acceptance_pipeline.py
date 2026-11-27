"""Run only metadata acceptance after the authorized CPU replay exits zero."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('launch', 'worker'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    root, record = args.root.resolve(strict=True), args.record.resolve(strict=True)
    assert record.is_relative_to(root)
    code = root / 'code_a10'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    if args.mode == 'launch':
        sources = {name: sha(record / name) for name in
                   ('acceptance_pipeline.py', 'full_replay_audit.py', 'postflight.py', 'r5_abn.jsonl.gz')}
        assert sources['full_replay_audit.py'] == '1135e11856bad024c329736c21c4c31049b58e19e527303397788a0e35174dec'
        assert sources['r5_abn.jsonl.gz'] == '1118238c6a87c19bd315a678b7a4c8ad8d3db01b7e72a8f9f27437c7e2e2e8f1'
        with (record / 'acceptance_pipeline.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                        'worker', '--root', str(root), '--record', str(record)],
                                       cwd=code, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        value = {'pid': process.pid, 'source_sha256': sources,
                 'started_unix': time.time(), 'GPU_budget': 0}
        write(record / 'acceptance_launch.json', value)
        print(json.dumps(value), flush=True)
        return 0
    try:
        while not (record / 'exit.json').exists():
            time.sleep(5)
        assert json.loads((record / 'exit.json').read_text())['returncode'] == 0, 'CPU_REPLAY_FAILED'
        launch = json.loads((record / 'acceptance_launch.json').read_text())
        for name, expected in launch['source_sha256'].items():
            assert sha(record / name) == expected, 'ACCEPTANCE_SOURCE_CHANGED'
        commands = [
            ('coverage', [sys.executable, '-B', '-m', 'tools.exp18.coverage_report',
                          '--replay-root', str(root / 'full_a10/replay'),
                          '--output', str(record / 'coverage'),
                          '--historical-steps', str(record / 'r5_abn.jsonl.gz')]),
            ('full_audit', [sys.executable, '-B', str(record / 'full_replay_audit.py'),
                           '--code-root', str(code), '--replay-root', str(root / 'full_a10/replay'),
                           '--coverage-root', str(record / 'coverage'), '--output', str(record / 'full_audit')]),
            ('postflight', [sys.executable, '-B', str(record / 'postflight.py'),
                           '--root', str(root), '--record', str(record)]),
        ]
        exits = []
        for name, command in commands:
            started = time.time()
            with (record / (name + '.log')).open('xb') as log:
                process = subprocess.Popen(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
                returncode = process.wait()
            row = {'phase': name, 'command': command, 'returncode': returncode,
                   'seconds': time.time() - started, 'log_sha256': sha(record / (name + '.log'))}
            write(record / (name + '_exit.json'), row)
            exits.append(row)
            print(json.dumps({'phase': name, 'returncode': returncode, 'seconds': row['seconds']}), flush=True)
            assert returncode == 0, name + '_FAILED'
        write(record / 'acceptance_exit.json', {'status': 'PASS_METADATA_AND_POSTFLIGHT',
                                               'phases': exits, 'GPU_budget': 0,
                                               'finished_unix': time.time()})
        return 0
    except BaseException as error:
        write(record / 'acceptance_failure.json', {'status': 'HOLD', 'error': repr(error),
                                                  'preserve_all_outputs': True, 'GPU_budget': 0})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
