"""Single-writer, durable journal for EXP18 V2 checkpoint intervals."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

from .common import digest, require
from .storage_policy import BATCHES, INTERVAL, MAX_REPLAY_UPDATES, VAL_JOBS

ZERO = "0" * 64


def fsync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def lease(root, *, create):
    """Kernel releases the lock on process exit. The lease file is permanent."""
    path = root / "run.lock"
    if create:
        with path.open("xb") as stream:
            stream.write(b"EXP18_STORAGE_V2\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(root)
    require(path.is_file() and not path.is_symlink(), "Missing or unsafe lease")
    with path.open("rb") as stream:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Closing this handle releases either platform lock.
        yield


class StorageJournal:
    def __init__(self, path, bindings, policy, *, create=False):
        self.path, self.bindings, self.policy = Path(path), digest(bindings), policy
        self.identity = {k: bindings[k] for k in ("execution_id", "arm_id")}
        self.final_update = policy["permanent_epochs"][-1] * BATCHES
        self.rows, self.bytes = [], 0
        self.run_attempt = None
        self.pending_step, self.pending_val, self.pending_retire, self.pending_recovery = None, None, None, None
        self.executed, self.persisted = 0, 0
        self.complete = False
        self.checkpoints, self.retired, self.val = {}, set(), {}
        self.tail_steps, self.expected_replay = {}, {}
        self.diagnostics, self.best = {}, {}
        self.poisoned = False
        if create:
            with self.path.open("xb") as stream:
                os.fsync(stream.fileno())
            fsync_directory(self.path.parent)
        require(self.path.is_file() and not self.path.is_symlink(), "Missing or unsafe journal")
        with self.path.open("rb") as stream:
            for line in stream:
                require(line.endswith(b"\n"), "Torn journal line")
                row = json.loads(line)
                require(set(row) == {"sequence", "previous", "event", "bindings", "identity", "fields", "sha256"}
                        and row["identity"] == self.identity,
                        "Invalid journal row schema")
                require(row["sequence"] == len(self.rows) and row["previous"] == self.prefix["sha256"]
                        and row["bindings"] == self.bindings
                        and row["sha256"] == digest({k: v for k, v in row.items() if k != "sha256"}),
                        "Journal chain or identity mismatch")
                self._apply(row)
                self.rows.append(row)
                self.bytes += len(line)

    @property
    def committed(self):
        return self.persisted

    @property
    def prefix(self):
        return {"rows": len(self.rows), "sha256": self.rows[-1]["sha256"] if self.rows else ZERO, "bytes": self.bytes}

    @property
    def boundary_evaluation_complete(self):
        return (self.persisted == 0 or all((self.persisted, job) in self.val for job in VAL_JOBS)
                and self.persisted in self.diagnostics and self.persisted in self.best)

    @property
    def rolling(self):
        return [step for step, spec in self.checkpoints.items() if spec["kind"] == "rolling" and step not in self.retired]

    def _apply(self, row):
        event, fields = row["event"], row["fields"]
        step = fields.get("step")
        attempt = fields.get("attempt")
        require(not self.complete or event in ("RETIRE_BEGIN", "RETIRE_COMMIT"),
                "A completed run cannot accept training/evaluation")
        if self.pending_val is not None:
            require(event == "VALIDATION_COMMIT", "VAL attempt has no commit")
        if self.pending_retire is not None:
            require(event == "RETIRE_COMMIT", "Retirement attempt has no commit")
        if self.pending_recovery is not None:
            require(event == "RECOVERY_COMPLETE", "Recovery attempt has no commit")
        if self.pending_step is not None:
            require(event in ("UPDATE_EXECUTED", "SEGMENT_ABANDONED"), "Previous update is unfinished")

        if event == "RUN_BEGIN":
            require(not self.rows and isinstance(attempt, str) and attempt, "Run already initialized")
            self.run_attempt = attempt
        elif event == "ATTEMPT_BEGIN":
            require(self.checkpoints and self.boundary_evaluation_complete and len(self.rolling) <= 2
                    and attempt == self.run_attempt and step == self.executed + 1 <= self.final_update
                    and step <= self.persisted + MAX_REPLAY_UPDATES, "Update is out of order or over budget")
            self.pending_step = step
        elif event == "UPDATE_EXECUTED":
            require(step == self.pending_step and attempt == self.run_attempt, "Update has no matching reservation")
            evidence = fields["evidence"]
            require(evidence["global_update"] == step and evidence["epoch"] == (step - 1) // BATCHES + 1
                    and evidence["batch_position"] == (step - 1) % BATCHES + 1,
                    "Update evidence cursor differs")
            identity = fields["replay_sha256"]
            require(evidence["checkpoint_anchor_sha256"] == self.checkpoints[self.persisted]["receipt"]["sha256"]
                    and identity == digest({k: v for k, v in evidence.items()
                                            if k not in ("resource", "checkpoint_anchor_sha256")}),
                    "Update evidence fingerprint differs")
            expected = self.expected_replay.get(step)
            require(expected is None or identity == expected, "Restarted step differs from its abandoned evidence")
            self.tail_steps[step] = identity
            self.executed, self.pending_step = step, None
        elif event == "CHECKPOINT_COMMIT":
            require(self.run_attempt and attempt == self.run_attempt and fields["prefix"] == self.prefix
                    and step == self.executed and step not in self.checkpoints and step % (INTERVAL * BATCHES) == 0
                    and ((step == 0 and not self.checkpoints) or
                         (step > 0 and step == self.persisted + INTERVAL * BATCHES and self.boundary_evaluation_complete)),
                    "Checkpoint cursor, cadence or journal anchor differs")
            expected = "permanent" if step // BATCHES in self.policy["permanent_epochs"] else "rolling"
            require(fields["kind"] == expected, "Checkpoint retention kind differs")
            self.checkpoints[step] = fields
            self.persisted = step
            self.tail_steps.clear()
            self.expected_replay = {pos: value for pos, value in self.expected_replay.items() if pos > step}
        elif event == "DIAGNOSTIC_COMMIT":
            require(step == self.executed == self.persisted and step > 0 and step not in self.diagnostics
                    and fields["checkpoint_sha256"] == self.checkpoints[step]["receipt"]["sha256"],
                    "Diagnostic cursor/identity differs")
            self.diagnostics[step] = fields["artifact"]
        elif event in ("VALIDATION_BEGIN", "VALIDATION_COMMIT"):
            job = fields["job"]
            require(step == self.executed == self.persisted and step > 0 and step in self.diagnostics
                    and fields["checkpoint_sha256"] == self.checkpoints[step]["receipt"]["sha256"]
                    and job in VAL_JOBS and (step, job) not in self.val, "Validation cursor/identity differs")
            if event == "VALIDATION_BEGIN":
                require(self.pending_val is None, "Validation already started")
                self.pending_val = (step, job)
            else:
                require(self.pending_val == (step, job), "Validation has no reservation")
                self.val[step, job] = fields
                self.pending_val = None
        elif event == "BEST_COMMIT":
            require(step == self.persisted == self.executed and step not in self.best
                    and all((step, job) in self.val for job in VAL_JOBS)
                    and set(fields["selections"]) == {"MASTER", "EMA"}, "Best selection cursor differs")
            self.best[step] = fields["selections"]
        elif event == "SEGMENT_ABANDONED":
            require(attempt == self.run_attempt and fields["anchor"] == self.persisted
                    and step == self.executed and fields["pending_step"] == self.pending_step
                    and (self.executed > self.persisted or self.pending_step is not None)
                    and self.executed - self.persisted <= MAX_REPLAY_UPDATES, "Invalid abandoned segment")
            self.expected_replay.update(self.tail_steps)
            self.tail_steps.clear()
            self.executed, self.pending_step = self.persisted, None
        elif event == "RECOVERY_BEGIN":
            require(self.checkpoints and step == self.persisted == self.executed and attempt != self.run_attempt
                    and fields["previous_attempt"] == self.run_attempt and isinstance(attempt, str) and bool(attempt),
                    "Recovery anchor/attempt differs")
            self.pending_recovery = attempt
        elif event == "RECOVERY_COMPLETE":
            require(attempt is not None and attempt == self.pending_recovery and step == self.persisted
                    and fields["checkpoint_sha256"] == self.checkpoints[step]["receipt"]["sha256"],
                    "Recovery has no reservation")
            self.run_attempt, self.pending_recovery = attempt, None
        elif event == "RETIRE_BEGIN":
            require(self.rolling and step == min(self.rolling) and (self.complete or len(self.rolling) == 3)
                    and fields["artifact_sha256"] == self.checkpoints[step]["receipt"]["sha256"],
                    "Only the oldest eligible owned rolling checkpoint can retire")
            self.pending_retire = step
        elif event == "RETIRE_COMMIT":
            require(step == self.pending_retire and fields["artifact_sha256"] == self.checkpoints[step]["receipt"]["sha256"],
                    "Retirement lacks its reservation")
            self.retired.add(step)
            self.pending_retire = None
        elif event == "TRAIN_COMPLETE":
            require(step == self.persisted == self.final_update and self.boundary_evaluation_complete
                    and set(fields["endpoints"]) == {"MASTER", "EMA"}, "Final artifacts or training incomplete")
            self.complete = True
        else:
            raise ValueError("Unknown journal event: " + event)

    def append(self, event, **fields):
        require(not self.poisoned and self.path.stat().st_size == self.bytes, "Journal writer changed; reopen under lease")
        row = {"sequence": len(self.rows), "previous": self.prefix["sha256"], "event": event,
               "bindings": self.bindings, "identity": self.identity, "fields": fields}
        row["sha256"] = digest(row)
        line = (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.poisoned = True
        self._apply(row)
        with self.path.open("ab") as stream:
            require(stream.write(line) == len(line), "Journal write was short")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        self.bytes += len(line)
        self.poisoned = False
        return row


def attempt_id():
    return uuid.uuid4().hex
