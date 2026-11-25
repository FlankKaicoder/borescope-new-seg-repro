"""Stage-separated command line. CPU mode never dispatches real-network code."""
from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
from pathlib import Path

from .common import require, source_manifest, write_json_exclusive
from .protocol import FORBIDDEN_CPU_OPERATIONS, make_protocol, require_operation, validate_protocol


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("audit", "replay", "cpu-test", *FORBIDDEN_CPU_OPERATIONS))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int, choices=(100, 200))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.config:
        from .runtime_config import load_cpu_profile
        _, protocol = load_cpu_profile(args.config, root)
        require(args.protocol is None and (args.epochs is None or args.epochs == protocol["train"]["epochs"]),
                "Conflicting runtime/protocol/epoch selections")
    else:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol else make_protocol(args.epochs or 200)
    validate_protocol(protocol)
    # This happens BEFORE Torch, native adapters, checkpoint loaders, or CUDA imports.
    require_operation(args.operation, protocol)
    require(args.output is not None, "CPU operation needs an independent output directory")
    output = args.output.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    else:
        require(output.is_dir(), "Resume output does not exist")
    from .guards import CPUExecutionGuard, configure_cpu_environment
    configure_cpu_environment(output)
    started = time.monotonic()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    test_root = root / "private_assets/exp18_tests"
    test_root.mkdir(parents=True, exist_ok=True)
    guard = CPUExecutionGuard((output, test_root), raw_roots=("/root/autodl-tmp/损伤训练数据集",))
    with guard:
        if args.operation == "audit":
            from .data import load_runtime_config
            result = {"config": load_runtime_config(), "source": source_manifest(root), "protocol": protocol}
        elif args.operation == "cpu-test":
            suite = unittest.defaultTestLoader.discover(str(root / "tools/exp18/tests"),
                                                       pattern="test_*.py", top_level_dir=str(root))
            with (output / "unittest.log").open("x", encoding="utf-8") as log:
                report = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
            result = {"tests_run": report.testsRun, "failures": len(report.failures), "errors": len(report.errors),
                      "skipped": len(report.skipped), "success": report.wasSuccessful()}
        else:
            from .data import load_runtime_config
            from .replay import run_replay
            def progress(value):
                print(json.dumps(value, ensure_ascii=False), flush=True)
            result = run_replay(load_runtime_config(), protocol, output / "replay",
                                max_epochs=args.max_epochs, resume=args.resume, progress=progress)
        try:
            import resource
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            peak_rss = None
        report = {"operation": args.operation, "result": result,
                  "seconds": time.monotonic() - started,
                  "peak_RSS_KiB": peak_rss,
                  "guard": guard.receipt()}
        # A resumed operation receives a new immutable receipt instead of replacing one.
        receipt = output / f"receipt_{time.time_ns()}.json"
        write_json_exclusive(receipt, report)
    print(json.dumps({"receipt": str(receipt), "seconds": report["seconds"],
                      "success": result.get("success", result.get("status", "COMPLETE"))}, ensure_ascii=False), flush=True)
    if args.operation == "cpu-test" and not result["success"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
