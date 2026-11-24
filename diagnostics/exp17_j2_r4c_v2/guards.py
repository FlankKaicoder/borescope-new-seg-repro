"""Pure-Python dispatch and filesystem guards; no model or tensor operations."""
import os
from pathlib import Path
from types import MethodType


def attach_method_counter(model, counts, permitted):
    owned = 'forward' in model.__dict__
    original_attribute = model.__dict__.get('forward')
    original = model.forward

    def counted(_model, value, *args, **kwargs):
        kind = 'dict' if isinstance(value, dict) else 'tensor'
        if not permitted():
            raise RuntimeError('Forward method outside the claimed A traversal')
        position = len(counts['method_dispatch'])
        if position >= 2 or kind != ('dict', 'tensor')[position]:
            raise RuntimeError('Extra or misordered forward method dispatch')
        counts['method_dispatch'].append(kind)
        return original(value, *args, **kwargs)

    model.forward = MethodType(counted, model)

    def restore():
        if owned:
            model.forward = original_attribute
        else:
            del model.forward
    return restore


def validate_dispatch(counts):
    expected = {'A': list(range(24)), 'N': list(range(17)), 'projection': 2,
                'root_dispatch': ['dict'], 'method_dispatch': ['dict', 'tensor']}
    if counts != expected:
        raise RuntimeError('Dispatch mismatch: expected='+repr(expected)+' observed='+repr(counts))
    return expected


def resolve_at(path, dir_fd=None, *, fd_path=None):
    value = Path(os.fsdecode(path))
    if not value.is_absolute() and dir_fd is not None and dir_fd != -1:
        resolver = fd_path or (lambda fd: os.readlink('/proc/self/fd/'+str(fd)))
        value = Path(resolver(dir_fd))/value
    return value.resolve()


def audit_policy(root, event, arguments, *, fd_path=None):
    root = Path(root).resolve()
    if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
        path, mode, flags = arguments
        writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                   or bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if writing and not resolve_at(path).is_relative_to(root):
            raise RuntimeError('Write outside independent diagnostic directory: '+os.fsdecode(path))
    if event in ('os.remove', 'os.rmdir'):
        path, dir_fd = arguments
        targets = [resolve_at(path, dir_fd, fd_path=fd_path)]
    elif event == 'os.rename':
        source, destination, source_fd, destination_fd = arguments
        targets = [resolve_at(source, source_fd, fd_path=fd_path),
                   resolve_at(destination, destination_fd, fd_path=fd_path)]
    else:
        targets = []
    if any(not p.is_relative_to(root/'settings') for p in targets):
        raise RuntimeError('Deletion/rename outside disposable settings: '+repr(targets))


def seal_before_check(write_record, validate, record):
    """Persist observations even when the following acceptance check fails."""
    write_record(record)
    return validate()
