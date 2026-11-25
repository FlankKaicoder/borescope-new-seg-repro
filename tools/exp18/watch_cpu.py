"""Small stdlib supervisor retaining actual exit status and cgroup samples."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    args.record.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MALLOC_ARENA_MAX='1')
    command = [sys.executable, '-B', '-m', 'tools.exp18', 'replay', '--output', str(args.output)]
    if args.max_epochs: command += ['--max-epochs', str(args.max_epochs)]
    if args.config: command += ['--config', str(args.config)]
    with (args.record / 'stdout.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=args.code, env=env, stdout=log, stderr=subprocess.STDOUT)
    started = time.time()
    with (args.record / 'process.json').open('x') as f:
        json.dump({'pid': process.pid, 'command': command, 'started_unix': started}, f)
    with (args.record / 'resource_samples.jsonl').open('x') as f:
        while process.poll() is None:
            row = {'unix': time.time()}
            try:
                text = Path(f'/proc/{process.pid}/status').read_text()
                row['status'] = {k: v.strip() for line in text.splitlines() if ':' in line
                                 for k, v in [line.split(':', 1)] if k in ('State', 'VmRSS', 'VmHWM', 'VmSize', 'Threads')}
                row['cgroup_current'] = int(Path('/sys/fs/cgroup/memory.current').read_text())
                row['cgroup_events'] = Path('/sys/fs/cgroup/memory.events').read_text()
            except FileNotFoundError: pass
            f.write(json.dumps(row) + '\n'); f.flush()
            time.sleep(1)
    with (args.record / 'exit.json').open('x') as f:
        json.dump({'returncode': process.returncode, 'seconds': time.time() - started}, f)
    print(json.dumps({'returncode': process.returncode, 'record': str(args.record)}), flush=True)
    return 0 if process.returncode == 0 else 1


if __name__ == '__main__': raise SystemExit(main())
