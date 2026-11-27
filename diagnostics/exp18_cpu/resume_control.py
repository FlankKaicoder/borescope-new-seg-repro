"""CPU-only resume controls outside the immutable a10 execution snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/exp18_cpu/20261125T131451Z')
CODE = ROOT / 'code_a10'
PYTHON = '/root/autodl-tmp/borescope-new-seg-repro/.venv/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def resources():
    return {name: Path('/sys/fs/cgroup', name).read_text().strip()
            for name in ('cpu.max', 'memory.max', 'memory.current', 'memory.events')}


def preflight(record):
    baseline_path = ROOT / 'closure/readonly_input_and_source_postcheck.json'
    baseline = json.loads(baseline_path.read_text())
    manifest_path = record / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert len(baseline['inputs']) == 1899
    assert len(manifest['files']) == 446
    checked = []
    for row in baseline['inputs']:
        path = Path(row['path'])
        assert 'test' not in {part.lower() for part in path.parts}
        actual = sha(path)
        assert actual == row['sha256'], ('INPUT_CHANGED', str(path))
        checked.append({'path': str(path), 'bytes': path.stat().st_size,
                        'mtime_ns': path.stat().st_mtime_ns, 'sha256': actual})
    for row in manifest['files']:
        path = CODE / row['path']
        assert sha(path) == row['sha256'], ('SOURCE_CHANGED', row['path'])
    assert sha(ROOT / 'source_a10.tar') == manifest['archive_sha256']
    sys.path.insert(0, str(CODE))
    from tools.exp18.replay import replay_source_manifest
    from tools.exp18.runtime_config import load_cpu_profile
    runtime, protocol = load_cpu_profile(CODE / 'configs/exp18/runtime_h200.json', CODE)
    assert replay_source_manifest() == baseline['runtime_source'], 'NATIVE_ENVIRONMENT_CHANGED'
    import importlib.metadata
    torch_dist = importlib.metadata.distribution('torch')
    for name, expected in baseline['additional_Torch_sources'].items():
        assert sha(torch_dist.locate_file('torch/' + name)) == expected, name
    assert 'torch' not in sys.modules, 'PREFLIGHT_MUST_NOT_IMPORT_TORCH'
    resource = resources()
    assert int(resource['memory.max']) >= 4 * 1024**3
    disk = shutil.disk_usage(ROOT)
    assert disk.free >= 5 * 1024**3
    value = {'status': 'PASS', 'checked_inputs': checked, 'input_files': 1899,
             'source_files': 446, 'baseline_receipt_sha256': sha(baseline_path),
             'source_manifest_sha256': sha(manifest_path), 'resources': resource,
             'disk_free_bytes': disk.free, 'runtime': runtime,
             'protocol': protocol, 'torch_imported': False,
             'real_model_constructions': 0, 'GPU_calls': 0, 'VAL_passes': 0,
             'TEST_accesses': 0, 'checked_at_unix': time.time()}
    write(record / 'preflight.json', value)
    print(json.dumps({'status': 'PASS', 'input_files': 1899, 'source_files': 446,
                      'resources': resource, 'disk_free_bytes': disk.free}), flush=True)


def launch(record):
    prior = json.loads((record / 'preflight.json').read_text())
    assert prior['status'] == 'PASS'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MALLOC_ARENA_MAX='1', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    with (record / 'supervisor.log').open('xb') as log:
        process = subprocess.Popen([PYTHON, '-B', str(Path(__file__).resolve()),
                                    'worker', '--record', str(record)],
                                   cwd=CODE, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    value = {'supervisor_pid': process.pid, 'started_unix': time.time(),
             'preflight_sha256': sha(record / 'preflight.json'),
             'control_sha256': sha(__file__), 'GPU_budget': 0}
    write(record / 'launch.json', value)
    print(json.dumps(value), flush=True)


def worker(record):
    command = [PYTHON, '-B', '-m', 'tools.exp18', 'replay',
               '--config', 'configs/exp18/runtime_h200.json',
               '--output', str(ROOT / 'full_a10'), '--resume']
    with (record / 'stdout.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
    started = time.time()
    write(record / 'process.json', {'pid': process.pid, 'command': command,
                                   'started_unix': started})
    with (record / 'resource_samples.jsonl').open('x') as stream:
        while process.poll() is None:
            sample = {'unix': time.time(), 'resources': resources()}
            status = Path(f'/proc/{process.pid}/status')
            if status.exists():
                sample['process'] = {
                    k: v.strip() for line in status.read_text().splitlines() if ':' in line
                    for k, v in [line.split(':', 1)]
                    if k in ('State', 'VmRSS', 'VmHWM', 'Threads')}
            stream.write(json.dumps(sample) + '\n')
            stream.flush()
            time.sleep(5)
    write(record / 'exit.json', {'returncode': process.returncode,
                                 'seconds': time.time() - started,
                                 'finished_unix': time.time(), 'GPU_budget': 0})
    return 0 if process.returncode == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('preflight', 'launch', 'worker'))
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    assert args.record.resolve().is_relative_to(ROOT)
    return {'preflight': preflight, 'launch': launch, 'worker': worker}[args.operation](args.record)


if __name__ == '__main__':
    raise SystemExit(main())
