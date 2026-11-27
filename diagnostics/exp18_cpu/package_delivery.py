"""Stream and verify the completed EXP18 CPU evidence without staging a server tar.

Only explicit EXP18 output trees are inventoried. Symlinks are archived as links,
never traversed; images and checkpoints behind links are never opened. Every
output is exclusive, and failures deliberately retain partial bytes.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import time


BASE = Path('/root/autodl-tmp/exp18_cpu')
DEFAULT_ROOT = BASE / '20261125T131451Z'
DEFAULT_RECORD = 'resume_20261128_01'
INVENTORY_NAME = 'artifact_inventory.json'
REPLAY_NAME = 'full_a10'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open_regular(path) as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def open_regular(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    stream = os.fdopen(descriptor, 'rb')
    try:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'Not a regular file: ' + str(path))
    except BaseException:
        stream.close()
        raise
    return stream


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def layout(root, record_name):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(BASE.resolve(strict=True)) and root != BASE, 'Root escaped EXP18 outputs')
    require(re.fullmatch(r'resume_[0-9]{8}_[0-9]{2}', record_name) is not None, 'Unsafe record name')
    for name in (REPLAY_NAME, record_name):
        path = root / name
        require(not path.is_symlink() and path.resolve(strict=True).parent == root and path.is_dir(),
                'Output tree escaped root or is a symlink: ' + name)
    return root, root / record_name


def member_name(value, record_name):
    require(isinstance(value, str) and '\\' not in value, 'Unsafe archive member')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '.' not in path.parts
            and path.parts and path.parts[0] in (REPLAY_NAME, record_name), 'Member escaped selected output trees')
    require(path.as_posix() == value, 'Noncanonical archive member')
    return path


def gate_json(root, path):
    path = Path(path)
    require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(root), 'Gate escaped outputs')
    with open_regular(path) as f:
        raw = f.read()
    return json.loads(raw), {'member': path.relative_to(root).as_posix(), 'sha256': digest_bytes(raw)}


def gates(root, record):
    summaries = []
    for path in sorted((root / REPLAY_NAME / 'replay').glob('attempt_*/summary.json')):
        value, binding = gate_json(root, path)
        if value.get('completed_epochs') == 200 and value.get('status') == 'CPU_DATA_READY_GPU_PENDING':
            summaries.append((value, binding))
    require(len(summaries) == 1, 'Exactly one completed 200-epoch replay is required')
    replay, replay_binding = summaries[0]
    require(replay['horizon'] == 200 and replay['counts']['batches'] == 4200
            and replay['counts']['images'] == 133600 and replay['counts']['raw_targets'] == 253200
            and replay['counts']['historical_verified'] == 2100 and len(replay['commits']) == 200,
            'Replay totals are incomplete')
    coverage, coverage_binding = gate_json(root, record / 'coverage/summary.json')
    require(coverage['status'] == 'CPU_COVERAGE_AUDIT_PASS_GPU_PENDING'
            and coverage['structural_gate'] == 'PASS' and coverage['audited_epochs'] == 200
            and coverage['audited_batches'] == 4200 and coverage['maximum_assignment_recomputed'] is True,
            'Full maximum-matching coverage audit is required')
    audit, audit_binding = gate_json(root, record / 'full_audit/summary.json')
    require(audit['status'] == 'PASS_FULL_200_EPOCH_4200_BATCH_METADATA'
            and audit['completed_epochs'] == 200 and audit['completed_batches'] == 4200
            and audit['coverage_maximum_recomputed'] is True
            and audit['independent_closed_cardinality']['batches_checked'] == 4200,
            'Independent full audit is required')
    postflight, postflight_binding = gate_json(root, record / 'postflight.json')
    require(postflight['schema'] == 'EXP18_CPU_POSTFLIGHT_V1' and postflight['status'] == 'PASS'
            and postflight['input_files'] == 1899 and postflight['source_files'] == 446
            and postflight['completed_epochs'] == 200 and postflight['completed_batches'] == 4200,
            'Final 1899-input/446-source postflight is required')
    require(postflight['coverage_summary_sha256'] == coverage_binding['sha256'],
            'Coverage summary differs from the postflight SHA binding')
    require(postflight['full_audit_summary_sha256'] == audit_binding['sha256'],
            'Independent audit summary differs from the postflight SHA binding')
    exit_receipt, exit_binding = gate_json(root, record / 'exit.json')
    require(exit_receipt['returncode'] == 0, 'CPU replay process did not exit successfully')
    binding = replay['bindings_sha256']
    require(binding == coverage['bindings_sha256'] == audit['bindings_sha256'], 'Gate bindings disagree')
    require(replay['CUDA_initialized'] is False and replay['lambda_state'] is None
            and all(replay[k] == 0 for k in ('real_model_constructions', 'real_model_forwards',
                'real_checkpoint_tensor_loads', 'VAL_passes', 'TEST_accesses')),
            'Replay exceeded CPU-only scope')
    return {'completed_epochs': 200, 'completed_batches': 4200, 'bindings_sha256': binding,
            'scientific_coverage_gate': coverage['scientific_coverage_gate'],
            'receipts': [replay_binding, coverage_binding, audit_binding, postflight_binding, exit_binding]}


def entry(root, path):
    info = path.lstat()
    result = {'member': path.relative_to(root).as_posix(), 'server_path': str(path),
              'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid, 'gid': info.st_gid,
              'mtime_ns': info.st_mtime_ns, 'device': info.st_dev, 'inode': info.st_ino,
              'hardlink_count': info.st_nlink}
    if stat.S_ISREG(info.st_mode):
        result.update(type='file', bytes=info.st_size, sha256=sha(path))
        after = path.lstat()
        require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during inventory')
    elif stat.S_ISLNK(info.st_mode):
        result.update(type='symlink', bytes=0, linkname=os.readlink(path))
    elif stat.S_ISDIR(info.st_mode):
        result.update(type='directory', bytes=0)
    else:
        result.update(type='excluded_special', bytes=0, file_type=stat.S_IFMT(info.st_mode))
    return result


def inventory(root, record_name):
    root, record = layout(root, record_name)
    inventory_path = record / INVENTORY_NAME
    require(not inventory_path.exists(), 'Inventory already exists; preserve it and use a new approved record')
    gate_record = gates(root, record)
    entries = []
    for tree in (root / REPLAY_NAME, record):
        entries.append(entry(root, tree))
        for directory, dirs, files in os.walk(tree, followlinks=False):
            directory = Path(directory)
            for name in sorted(dirs):
                path = directory / name
                item = entry(root, path)
                entries.append(item)
                if item['type'] != 'directory':
                    dirs.remove(name)
            for name in sorted(files):
                path = directory / name
                if path != inventory_path:
                    entries.append(entry(root, path))
    entries.sort(key=lambda r: r['member'])
    require(len({row['member'] for row in entries}) == len(entries), 'Duplicate inventory members')
    for row in entries:
        member_name(row['member'], record_name)
    value = {'schema': 'EXP18_CPU_DELIVERY_INVENTORY_V1', 'root': str(root), 'record_name': record_name,
             'selected_trees': [REPLAY_NAME, record_name], 'created_unix': time.time(),
             'gates': gate_record, 'entries': entries,
             'inventory_self_member': inventory_path.relative_to(root).as_posix(),
             'regular_files': sum(r['type'] == 'file' for r in entries),
             'symlinks': sum(r['type'] == 'symlink' for r in entries),
             'excluded_special': [r['member'] for r in entries if r['type'] == 'excluded_special'],
             'regular_source_bytes': sum(r['bytes'] for r in entries if r['type'] == 'file'),
             'tool_sha256': sha(Path(__file__))}
    write_json(inventory_path, value)
    print(json.dumps({'inventory': str(inventory_path), 'sha256': sha(inventory_path),
                      'regular_files': value['regular_files'], 'symlinks': value['symlinks'],
                      'bytes': value['regular_source_bytes']}), flush=True)


class HashingReader:
    def __init__(self, stream):
        self.stream, self.h, self.bytes = stream, hashlib.sha256(), 0

    def read(self, count=-1):
        block = self.stream.read(count)
        self.h.update(block); self.bytes += len(block)
        return block


def tar_info(row):
    info = tarfile.TarInfo(row['member'])
    for key in ('mode', 'uid', 'gid'):
        setattr(info, key, row[key])
    seconds, nanoseconds = divmod(row['mtime_ns'], 10**9)
    info.mtime = seconds
    info.pax_headers = {'mtime': f'{seconds}.{nanoseconds:09d}'}
    if row['type'] == 'file':
        info.type, info.size = tarfile.REGTYPE, row['bytes']
    elif row['type'] == 'symlink':
        info.type, info.linkname = tarfile.SYMTYPE, row['linkname']
    else:
        info.type = tarfile.DIRTYPE
    return info


def archive(root, record_name):
    root, record = layout(root, record_name)
    inventory_path = record / INVENTORY_NAME
    with open_regular(inventory_path) as stream:
        frozen = json.load(stream)
    require(frozen['root'] == str(root) and frozen['record_name'] == record_name, 'Wrong inventory root')
    require(gates(root, record) == frozen['gates'], 'Completion Gate bytes changed after inventory')
    rows = [r for r in frozen['entries'] if r['type'] != 'excluded_special']
    rows.append(entry(root, inventory_path))
    rows.sort(key=lambda r: r['member'])
    require(len({r['member'] for r in rows}) == len(rows), 'Duplicate archive paths')
    # gzip/tar output is binary stdout. All status and failures go to stderr.
    with gzip.GzipFile(filename='', mode='wb', fileobj=sys.stdout.buffer, compresslevel=1, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for row in rows:
                member_name(row['member'], record_name)
                path = root / row['member']
                current = path.lstat()
                require(stat.S_IMODE(current.st_mode) == row['mode']
                        and current.st_uid == row['uid'] and current.st_gid == row['gid'], 'File attributes changed')
                if row['type'] == 'directory':
                    require(stat.S_ISDIR(current.st_mode), 'Directory type changed')
                    tar.addfile(tar_info(row))
                elif row['type'] == 'symlink':
                    require(stat.S_ISLNK(current.st_mode) and os.readlink(path) == row['linkname']
                            and current.st_mtime_ns == row['mtime_ns'], 'Symlink changed')
                    tar.addfile(tar_info(row))
                else:
                    require(stat.S_ISREG(current.st_mode) and current.st_size == row['bytes']
                            and current.st_mtime_ns == row['mtime_ns'], 'Regular file changed')
                    with open_regular(path) as source:
                        read = HashingReader(source)
                        tar.addfile(tar_info(row), read)
                        require(read.bytes == row['bytes'] and read.h.hexdigest() == row['sha256'], 'Streamed SHA differs')
                    after = path.lstat()
                    require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
                            (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns), 'File changed during transfer')


def validate_inventory(frozen, raw):
    require(frozen['schema'] == 'EXP18_CPU_DELIVERY_INVENTORY_V1', 'Unknown inventory')
    record = frozen['record_name']
    require(re.fullmatch(r'resume_[0-9]{8}_[0-9]{2}', record) is not None, 'Unsafe record')
    require(frozen['selected_trees'] == [REPLAY_NAME, record], 'Unexpected archive trees')
    expected = {}
    for row in frozen['entries']:
        member_name(row['member'], record)
        require(row['member'] not in expected, 'Duplicate manifest member')
        if row['type'] != 'excluded_special':
            require(row['type'] in ('file', 'directory', 'symlink'), 'Unsupported member type')
            expected[row['member']] = row
    self_member = frozen['inventory_self_member']
    require(self_member == f'{record}/{INVENTORY_NAME}' and self_member not in expected, 'Unsafe self inventory')
    expected[self_member] = {'member': self_member, 'server_path': frozen['root'] + '/' + self_member,
                             'type': 'file', 'bytes': len(raw), 'sha256': digest_bytes(raw)}
    require(frozen['gates']['completed_epochs'] == 200 and frozen['gates']['completed_batches'] == 4200,
            'Cannot capture incomplete evidence')
    return expected


def capture(args):
    source = Path(args.inventory).resolve(strict=True)
    with open_regular(source) as stream:
        raw = stream.read()
    require(digest_bytes(raw) == args.inventory_sha256, 'Downloaded inventory SHA differs')
    frozen = json.loads(raw)
    expected = validate_inventory(frozen, raw)
    target = Path(args.target).resolve()
    allowed = Path('E:/autodlwenjian/exp18').resolve()
    require(target != allowed and target.is_relative_to(allowed), 'Delivery escaped authorized E drive folder')
    target.mkdir(parents=True, exist_ok=False)
    partial, final = target / 'core.tar.gz.partial', target / 'core.tar.gz'
    try:
        with (target / INVENTORY_NAME).open('xb') as stream:
            stream.write(raw)
        command = shlex.join([args.server_python, '-B', args.server_script, 'archive',
                              '--root', frozen['root'], '--record-name', frozen['record_name']])
        with partial.open('xb') as output, (target / 'transfer.log').open('xb') as error:
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.ssh, command],
                                       stdout=output, stderr=error)
            require(process.wait() == 0, 'SSH archive failed; partial file preserved')
            output.flush(); os.fsync(output.fileno())
        actual, seen = {}, set()
        with tarfile.open(partial, 'r:gz') as tar:
            for member in tar:
                name = member.name.rstrip('/')
                member_name(name, frozen['record_name'])
                require(name not in seen and name in expected, 'Duplicate or extra tar member')
                seen.add(name); wanted = expected[name]
                require((member.isfile(), member.isdir(), member.issym()) ==
                        (wanted['type'] == 'file', wanted['type'] == 'directory', wanted['type'] == 'symlink'),
                        'Tar member type changed')
                if 'mode' in wanted:
                    mtime_ns = int(Decimal(member.pax_headers.get('mtime', str(member.mtime))) * 10**9)
                    require(member.mode == wanted['mode'] and member.uid == wanted['uid']
                            and member.gid == wanted['gid'] and mtime_ns == wanted['mtime_ns'],
                            'Tar attributes differ from inventory')
                if member.isfile():
                    require(member.size == wanted['bytes'], 'Tar member size differs')
                    h, size = hashlib.sha256(), 0
                    with tar.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            h.update(block); size += len(block)
                    require(size == wanted['bytes'] and h.hexdigest() == wanted['sha256'], 'Tar member SHA differs')
                elif member.issym():
                    require(member.linkname == wanted['linkname'], 'Symlink target differs')
                actual[name] = dict(wanted)
                if 'mode' not in wanted:
                    # The manifest cannot hash-bind its own stat fields. Record
                    # the observed tar metadata without claiming that comparison.
                    actual[name].update(mode=member.mode, uid=member.uid, gid=member.gid,
                        mtime_ns=int(Decimal(member.pax_headers.get('mtime', str(member.mtime))) * 10**9))
        require(seen == set(expected), 'Missing tar members')
        # Tar iteration stops at its zero blocks and can leave the gzip trailer
        # unread. Drain the complete compressed stream separately so gzip checks
        # every CRC32/ISIZE trailer before a formal archive can be published.
        raw_uncompressed_bytes = 0
        with gzip.open(partial, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                raw_uncompressed_bytes += len(block)
        record = frozen['record_name']
        samples = [f'{record}/preflight.json', f'{record}/coverage/summary.json',
                   f'{record}/full_audit/steps.csv', f'{record}/stdout.log']
        require(all(name in actual and actual[name]['type'] == 'file' for name in samples), 'Four restore samples missing')
        restored = target / 'sample_restore'; restored.mkdir()
        with tarfile.open(partial, 'r:gz') as tar:
            for index, name in enumerate(samples):
                output = restored / f'{index:02d}_{PurePosixPath(name).name}'
                with tar.extractfile(name) as stream, output.open('xb') as dst:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        dst.write(block)
                require(sha(output) == actual[name]['sha256'], 'Sample restore SHA differs')
        index_path = target / 'files.csv.gz'
        with gzip.open(index_path, 'xt', encoding='utf-8', newline='') as stream:
            columns = ['experiment', 'server_path', 'archive', 'member', 'type', 'bytes', 'sha256',
                       'linkname', 'mode', 'uid', 'gid', 'mtime_ns', 'device', 'inode', 'hardlink_count']
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for name, row in sorted(actual.items()):
                writer.writerow({'experiment': 'EXP18_CPU_20261128', 'archive': final.name,
                                 **{key: row.get(key, '') for key in columns if key not in ('experiment', 'archive')}})
        package = {'file': final.name, 'bytes': partial.stat().st_size, 'sha256': sha(partial)}
        require(not final.exists(), 'Formal archive already exists')
        partial.rename(final)
        value = {'schema': 'EXP18_CPU_DELIVERY_VERIFICATION_V1',
                 'status': 'PASS_MEMBER_BYTES_LINKS_ATTRIBUTES_AND_SAMPLE_RESTORE',
                 'members': len(actual), 'regular_files': sum(r['type'] == 'file' for r in actual.values()),
                 'symlinks': sum(r['type'] == 'symlink' for r in actual.values()),
                 'archived_regular_bytes': sum(r['bytes'] for r in actual.values() if r['type'] == 'file'),
                 'excluded_special': frozen['excluded_special'], 'sample_restore_members': samples,
                 'inventory_sha256': digest_bytes(raw), 'package': package,
                 'gzip_full_stream_verified': True,
                 'raw_uncompressed_bytes': raw_uncompressed_bytes,
                 'files_index': {'file': index_path.name, 'sha256': sha(index_path)},
                 'gate_evidence': frozen['gates'], 'models_or_tensors_loaded': 0,
                 'server_files_deleted': 0, 'old_archives_changed': False,
                 'tool_sha256': sha(Path(__file__)),
                 'limits': ['Symlinks are preserved as link metadata, never followed or sample-extracted.',
                            'Hardlink inode relationships are recorded; regular files are stored as independent byte members.',
                            'Directories preserve snapshot attributes; inventory creation may update their live mtimes.',
                            'Inventory self-member bytes are verified; its tar stat fields are recorded without an independent pre-inventory binding.',
                            'This archive does not include immutable inputs or the earlier a10 source package.']}
        write_json(target / 'archive_verification.json', value)
        print(json.dumps({'status': value['status'], 'members': len(actual), 'target': str(target),
                          'package_sha256': package['sha256'], 'bytes': package['bytes']}), flush=True)
    except BaseException as error:
        write_json(target / 'capture_failure.json', {'status': 'HOLD', 'error': repr(error),
                                                     'partial_files_preserved': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    for operation in ('inventory', 'archive'):
        item = sub.add_parser(operation)
        item.add_argument('--root', type=Path, default=DEFAULT_ROOT)
        item.add_argument('--record-name', default=DEFAULT_RECORD)
    item = sub.add_parser('capture')
    item.add_argument('--inventory', type=Path, required=True)
    item.add_argument('--inventory-sha256', required=True)
    item.add_argument('--ssh', default='borescope7')
    item.add_argument('--server-python', default='/root/autodl-tmp/borescope-new-seg-repro/.venv/bin/python')
    item.add_argument('--server-script', required=True)
    item.add_argument('--target', type=Path, default=Path('E:/autodlwenjian/exp18/cpu-20261128'))
    args = parser.parse_args()
    if args.operation == 'capture':
        capture(args)
    elif args.operation == 'inventory':
        inventory(args.root, args.record_name)
    else:
        archive(args.root, args.record_name)


if __name__ == '__main__':
    main()
