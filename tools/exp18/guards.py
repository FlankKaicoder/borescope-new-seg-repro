"""CPU-phase I/O, real-model and CUDA initialization boundaries.

The audit hook is disabled when the context ends. IPC is transported as NumPy,
so no experiment file deletion or checkpoint rotation is needed by the loader.
"""
from __future__ import annotations

import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from .common import require, sha256

_ACTIVE = None


def active_guard():
    return _ACTIVE if _ACTIVE is not None and _ACTIVE.active else None


def configure_cpu_environment(output):
    output = Path(output).resolve()
    paths = {"YOLO_CONFIG_DIR": output / "runtime/ultralytics", "XDG_CACHE_HOME": output / "runtime/cache",
             "TORCH_HOME": output / "runtime/torch", "TMPDIR": output / "runtime/ipc",
             "TEMP": output / "runtime/ipc", "TMP": output / "runtime/ipc"}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false",
                      YOLO_OFFLINE="true", MPLCONFIGDIR=str(output / "runtime/matplotlib"))
    sys.dont_write_bytecode = True
    # Torch's optimizer lazily imports distributed JIT helpers that allocate a
    # stdlib TemporaryDirectory even for CPU AdamW. Preserve these files too.
    import tempfile
    tempfile.tempdir = str(paths["TMPDIR"])
    owned_temp = paths["TMPDIR"].resolve()

    class PersistentTemporaryDirectory:
        def __init__(self, suffix=None, prefix=None, dir=None, **_kwargs):
            directory = Path(dir).resolve() if dir is not None else owned_temp
            require(directory.is_relative_to(owned_temp), "Temporary directory escaped EXP18 IPC root")
            self.name = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=directory)

        def __enter__(self): return self.name
        def __exit__(self, *_exc): return None
        def cleanup(self): return None

    tempfile.TemporaryDirectory = PersistentTemporaryDirectory
    # Native Torch's zero-copy CPU tensor IPC uses an AF_UNIX resource-sharer
    # socket. Retain its pathname and directory instead of registering unlink
    # or rmtree finalizers. Anonymous shared storage is released by closing FDs.
    import multiprocessing.util as mp_util
    import multiprocessing.connection as mp_connection
    import uuid
    ipc_directories = {}
    def persistent_ipc_directory():
        pid = os.getpid()
        if pid not in ipc_directories:
            # resource_sharer adds its own random listener basename. Avoid an
            # extra directory component: Linux AF_UNIX addresses are <=107 B.
            require(len(os.fsencode(owned_temp)) + 25 < 108, "EXP18 IPC root is too long for AF_UNIX")
            ipc_directories[pid] = str(owned_temp)
        return ipc_directories[pid]
    mp_util.get_temp_dir = persistent_ipc_directory
    original_listener_init = mp_connection.SocketListener.__init__
    def retained_listener_init(listener, *args, **kwargs):
        original_listener_init(listener, *args, **kwargs)
        finalizer = getattr(listener, "_unlink", None)
        if finalizer is not None:
            finalizer.cancel()
            listener._unlink = None
    mp_connection.SocketListener.__init__ = retained_listener_init
    return {k: os.environ[k] for k in (*paths, "CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "YOLO_AUTOINSTALL")}


def _test_path(path):
    parts = {p.casefold() for p in Path(path).parts}
    return bool(parts & {"test", "testing_split", "test_split"})


class CPUExecutionGuard:
    def __init__(self, output_roots, *, raw_roots=(), allowed_raw_files=()):
        self.output_roots = tuple(Path(p).resolve() for p in output_roots)
        self.raw_roots = tuple(Path(p).resolve() for p in raw_roots)
        self.allowed_raw_files = {Path(p).resolve() for p in allowed_raw_files}
        self.loads = {}
        self.active = False
        self.events = {"blocked_CUDA": 0, "blocked_real_model": 0, "blocked_load": 0,
                       "blocked_write": 0, "blocked_TEST": 0, "blocked_delete": 0,
                       "verified_CPU_loads": 0}
        self.reads = set()
        self.writes = set()

    def allow_cpu_load(self, path, expected_sha256, *, kind):
        require(kind in ("fixed_TRAIN_CPU_batch", "synthetic_CPU_checkpoint"), "Unapproved tensor load kind")
        path = Path(path).resolve(strict=True)
        if kind == "synthetic_CPU_checkpoint":
            require(any(path.is_relative_to(root) for root in self.output_roots), "Synthetic checkpoint escaped output")
        require(sha256(path) == expected_sha256, "CPU tensor package hash differs")
        self.loads[str(path)] = {"sha256": expected_sha256, "kind": kind}

    def check_read(self, value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        text = os.fsdecode(value)
        if _test_path(text):
            self.events["blocked_TEST"] += 1
            raise PermissionError("EXP18 cannot open a TEST path: " + text)
        path = Path(text).resolve()
        if _test_path(path):
            self.events["blocked_TEST"] += 1
            raise PermissionError("EXP18 cannot follow a path into TEST: " + str(path))
        if any(path.is_relative_to(root) for root in self.raw_roots) and path not in self.allowed_raw_files:
            raise PermissionError("Raw data read is not in the frozen TRAIN allowlist: " + str(path))
        self.reads.add(str(path))

    def check_write(self, value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(value)).resolve()
        if not any(path.is_relative_to(root) for root in self.output_roots):
            self.events["blocked_write"] += 1
            raise PermissionError("EXP18 output escaped its owned roots: " + str(path))
        self.writes.add(str(path))

    def _audit(self, event, args):
        if not self.active:
            return
        if event == "open":
            name, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing:
                self.check_write(name)
                if isinstance(name, (str, bytes, os.PathLike)) and Path(os.fsdecode(name)).exists():
                    exclusive = isinstance(flags, int) and bool(flags & os.O_EXCL)
                    append = (isinstance(mode, str) and "a" in mode) or (isinstance(flags, int) and bool(flags & os.O_APPEND))
                    if not exclusive and not append:
                        self.events["blocked_write"] += 1
                        raise PermissionError("EXP18 existing artifact cannot be overwritten: " + os.fsdecode(name))
            else:
                self.check_read(name)
        elif event in ("os.remove", "os.rmdir", "os.rename", "os.truncate"):
            self.events["blocked_delete"] += 1
            raise PermissionError("EXP18 artifacts cannot be deleted, replaced, or truncated: " + event)
        elif event in ("os.mkdir", "os.chmod", "os.link", "os.symlink"):
            self.check_write(args[0] if event not in ("os.link", "os.symlink") else args[1])

    def __enter__(self):
        global _ACTIVE
        import builtins
        import uuid
        import torch
        require(not torch.cuda.is_initialized(), "CUDA was initialized before EXP18 CPU guard")
        require(_ACTIVE is None, "Nested CPU guard is not supported")
        self.stack = ExitStack()
        guard = self
        original_to, original_module_init, original_load = torch.Tensor.to, torch.nn.Module.__init__, torch.load
        original_open = builtins.open
        settings_versions = {}

        def immutable_settings_open(file, mode="r", *args, **kwargs):
            # Pinned JSONDict saves its process settings repeatedly. Retain each
            # owned snapshot and preserve logical read-after-write semantics.
            if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).name == "settings.json":
                path = Path(os.fsdecode(file)).resolve()
                owned = any(path.is_relative_to(root / "runtime/ultralytics") for root in guard.output_roots)
                if owned and path.name == "settings.json":
                    if "w" in mode and path.exists():
                        target = path.with_name("settings_version_" + uuid.uuid4().hex + ".json")
                        stream = original_open(target, mode.replace("w", "x"), *args, **kwargs)
                        settings_versions[str(path)] = target
                        return stream
                    if "r" in mode and str(path) in settings_versions:
                        file = settings_versions[str(path)]
            return original_open(file, mode, *args, **kwargs)

        def no_cuda(*args, **kwargs):
            guard.events["blocked_CUDA"] += 1
            raise PermissionError("EXP18 CPU phase forbids CUDA initialization/transfer")

        def cpu_to(tensor, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            if isinstance(device, (str, torch.device)) and str(device).startswith("cuda"):
                return no_cuda()
            return original_to(tensor, *args, **kwargs)

        def module_init(module, *args, **kwargs):
            if type(module).__module__.startswith(("ultralytics.", "transformers.")):
                guard.events["blocked_real_model"] += 1
                raise PermissionError("EXP18 CPU phase forbids real model construction")
            return original_module_init(module, *args, **kwargs)

        def cpu_load(file, *args, **kwargs):
            if not isinstance(file, (str, os.PathLike)):
                guard.events["blocked_load"] += 1
                raise PermissionError("Unbound tensor stream load forbidden")
            name = str(Path(file).resolve())
            record = guard.loads.get(name)
            if record is None or sha256(name) != record["sha256"]:
                guard.events["blocked_load"] += 1
                raise PermissionError("Unapproved real checkpoint/tensor package: " + name)
            # Never accept a caller-supplied CUDA map.
            require(not args, "CPU tensor loading requires explicit keyword arguments")
            kwargs["map_location"] = "cpu"
            guard.events["verified_CPU_loads"] += 1
            return original_load(file, **kwargs)

        for owner, name, replacement in ((torch.cuda, "_lazy_init", no_cuda),
                                          (torch.Tensor, "cuda", no_cuda),
                                          (torch.nn.Module, "cuda", no_cuda),
                                          (torch.Tensor, "to", cpu_to),
                                          (torch.nn.Module, "__init__", module_init),
                                          (torch, "load", cpu_load),
                                          (builtins, "open", immutable_settings_open)):
            self.stack.enter_context(patch.object(owner, name, replacement))
        sys.addaudithook(self._audit)
        self.active = True
        _ACTIVE = self
        return self

    def __exit__(self, *exc):
        global _ACTIVE
        self.active = False
        _ACTIVE = None
        import torch
        initialized = torch.cuda.is_initialized()
        self.stack.close()
        require(not initialized, "CUDA became initialized in CPU phase")

    def receipt(self):
        return {"events": dict(self.events), "read_files": sorted(self.reads), "written_files": sorted(self.writes),
                "real_model_constructions": 0, "CUDA_initializations": 0,
                "real_checkpoint_tensor_loads": 0, "VAL_passes": 0, "TEST_reads": 0}
