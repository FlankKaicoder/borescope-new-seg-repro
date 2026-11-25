"""Launch a CPU-only task with exclusive logs; no existing files are replaced."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--operation", choices=("audit", "replay", "cpu-test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    args.record_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
    command = [sys.executable, "-B", "-m", "tools.exp18.remote_supervise",
               str(args.record_root), args.operation, str(args.output)]
    if args.max_epochs is not None: command += ["--max-epochs", str(args.max_epochs)]
    with (args.record_root / "stdout.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=args.code, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    info = {"pid": process.pid, "started_unix": time.time(), "command": command,
            "code": str(args.code), "output": str(args.output), "GPU_budget": 0}
    with (args.record_root / "launch.json").open("x") as stream:
        json.dump(info, stream, indent=2)
    print(json.dumps(info))


if __name__ == "__main__": main()
