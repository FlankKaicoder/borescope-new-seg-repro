"""Write an immutable exit receipt around an authorized CPU command."""
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    record, operation, output = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    from tools.exp18.__main__ import main as cpu_main
    started = time.time()
    code, error = 1, None
    try:
        code = int(cpu_main([operation, "--output", output, *sys.argv[4:]]))
    except BaseException as exc:
        error = repr(exc)
        traceback.print_exc()
    with (record / "exit.json").open("x", encoding="utf-8") as stream:
        json.dump({"exit_code": code, "error": error, "started_unix": started,
                   "finished_unix": time.time(), "GPU_budget": 0}, stream, indent=2)
    return code


if __name__ == "__main__": raise SystemExit(main())
