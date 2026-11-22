"""Identity-bound authorization and a permanent single-experiment update journal.

The public frozen protocol is deliberately non-authorizing. Only a separate
private ChatGPT Gate record can enable a future run. No command-line enable flag
exists. The application control anchor is independent of the output directory.
"""
import hashlib
import json
import os
from pathlib import Path

EXPERIMENT = "EXP17_J2_SEED42_FORMAL_V1"
AUTH_STAGE = "EXP17_G1_FORMAL_TRAIN_AUTHORIZATION"
ACTIONS = ["train", "scheduled_val_master_and_ema", "isolated_train_diagnostics", "retire_owned_ephemeral_epoch_cache"]
CAP = 2100


class HardGate(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise HardGate(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())


def control_anchor():
    return Path.home() / ".local" / "state" / "borescope" / EXPERIMENT


def source_catalog(repository):
    root = Path(repository)
    selected = set()
    # Include inherited helper dependencies too (e.g. the original Exp13 JSON
    # writer imported by Exp14), without rewriting any of their source bytes.
    for p in (root / "tools").rglob("*.py"):
        if "tests" not in p.parts and not p.name.startswith("test_"):
            selected.add(p)
    return {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(selected)}


def protocol_bindings(protocol, protocol_sha):
    return {"protocol_sha256": protocol_sha, "code_sha256": protocol["code"]["sha256"],
            "manifest_sha256": protocol["identity"]["trusted_manifest_sha256"],
            "native_identity_sha256": protocol["identity"]["native_train_identity_sha256"],
            "coco_sha256": protocol["identity"]["coco_sha256"],
            "seed": 42, "epochs": 100, "effective_updates": CAP,
            "environment": protocol["environment"]}


def validate_authorization(protocol_path, authorization_path, output, repository):
    # Check authorization first: a missing private Gate must not even import a
    # model, inspect a dataset, create output/settings, or claim an attempt.
    require(authorization_path is not None and Path(authorization_path).is_file(),
            "Separate private ChatGPT formal-training authorization is absent")
    p = load(protocol_path); a = load(authorization_path)
    require(p["status"] == "PENDING_CHATGPT_FORMAL_TRAINING_GATE" and p["formal_training_authorized"] is False,
            "Frozen public protocol status was changed")
    require(p.get("pending_fields") == [], "Formal protocol has unresolved audited fields")
    actual_sources = source_catalog(repository)
    require(actual_sources == p["code"]["files"] and digest(actual_sources) == p["code"]["sha256"],
            "Training source identity changed; a new protocol and Gate are required")
    require(a.get("stage") == AUTH_STAGE and a.get("gate") == "CHATGPT_FORMAL_TRAINING_GATE_PASS"
            and a.get("authorized") is True and a.get("experiment") == EXPERIMENT,
            "Invalid independent formal-training Gate")
    require(a.get("authorized_actions") == ACTIONS, "Gate must explicitly authorize the scheduled VAL and diagnostics too")
    require(a.get("bindings") == protocol_bindings(p, file_sha(protocol_path)), "Gate/protocol/data/code/budget identity differs")
    output = Path(output).absolute()
    require(not output.is_symlink() and str(output.resolve()) == a.get("output"), "Output is not the exact authorized location")
    require(a.get("control_anchor") == str(control_anchor().resolve()), "Permanent control anchor cannot be relocated")
    require(isinstance(a.get("execution_id"), str) and len(a["execution_id"]) >= 16, "Missing unique execution identity")
    return p, a


class RunLease:
    """OS process lease; a stale lock file is never treated as a new run."""
    def __init__(self, anchor):
        self.path = Path(anchor) / "process.lock"
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                if not self.stream.read(1): self.stream.write(b"0"); self.stream.flush()
                self.stream.seek(0); msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self.stream.close(); self.stream = None
            raise HardGate("Another executor owns the experiment process lease") from e
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0); msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


def claim_fresh(authorization, output):
    anchor, output = control_anchor(), Path(output)
    require(not anchor.exists(), "Permanent experiment claim already exists; output changes cannot reset it")
    require(not output.exists(), "Fresh output must not contain a prior run or copied execution seal")
    anchor.mkdir(parents=True, exist_ok=False)
    claim = {"experiment": EXPERIMENT, "authorization_sha256": digest(authorization),
             "execution_id": authorization["execution_id"], "output": str(output.resolve()), "cap": CAP}
    exclusive_json(anchor / "CLAIM.json", claim)
    output.mkdir(parents=True, exist_ok=False)
    exclusive_json(output / "EXECUTION_SEAL.json", claim)
    return anchor


def verify_existing_claim(authorization, output):
    anchor, output = control_anchor(), Path(output)
    expected = {"experiment": EXPERIMENT, "authorization_sha256": digest(authorization),
                "execution_id": authorization["execution_id"], "output": str(output.resolve()), "cap": CAP}
    require((anchor / "CLAIM.json").is_file() and (output / "EXECUTION_SEAL.json").is_file(),
            "Claim/seal missing; do not reconstruct or reset a run")
    require(load(anchor / "CLAIM.json") == load(output / "EXECUTION_SEAL.json") == expected,
            "Permanent and output execution seals disagree")
    require((anchor / "updates.jsonl").is_file(), "Claimed experiment ledger is missing; refuse budget reset")
    return anchor


class RunJournal:
    """Fsync'd hash chain; any uncommitted physical attempt is a hard stop."""
    def __init__(self, path, *, fresh=False):
        self.path = Path(path)
        if fresh:
            self.path.open("x", encoding="utf-8").close()
        require(self.path.is_file(), "Permanent update journal missing")
        self.rows = []
        self.committed = 0
        self.pending = None
        self.pending_aux = None
        previous = "0" * 64
        for line in self.path.read_text(encoding="utf-8").splitlines(keepends=True):
            require(line.endswith("\n"), "Truncated update journal")
            row = json.loads(line); stored = row.pop("sha256")
            require(row["sequence"] == len(self.rows) + 1 and row["previous"] == previous and digest(row) == stored,
                    "Journal hash/order identity conflict")
            row["sha256"] = stored; self._consume(row); self.rows.append(row); previous = stored
        require(self.pending is None, "AMBIGUOUS_INTERRUPTED_UPDATE: preserve evidence; independent recovery Gate required")
        require(self.pending_aux is None, "INTERRUPTED_EPOCH_EVENT: no automatic VAL/diagnostic retry")

    def _consume(self, row):
        if row["event"] == "ATTEMPT_BEGIN":
            step = row["step"]
            require(self.pending is None and self.pending_aux is None and step == self.committed + 1 and step <= CAP,
                    "Unexpected or excessive optimizer attempt")
            self.pending = step
        elif row["event"] == "UPDATE_COMMIT":
            require(self.pending == row["step"] and len(row.get("checkpoint_sha256", "")) == 64,
                    "Optimizer/checkpoint commit does not match its reserved update")
            self.committed = self.pending; self.pending = None
        elif row["event"] == "AUX_BEGIN":
            require(self.pending is None and self.pending_aux is None, "Concurrent epoch event or optimizer update")
            self.pending_aux = row["epoch"]
        elif row["event"] == "AUX_COMMIT":
            require(self.pending_aux == row["epoch"] and self.pending is None, "Unmatched epoch event commit")
            self.pending_aux = None

    def append(self, event, **values):
        row = {"sequence": len(self.rows) + 1, "previous": self.rows[-1]["sha256"] if self.rows else "0" * 64,
               "event": event, **values}
        # Validate on a disposable copy before persisting; invalid records must
        # not mutate the actual audit state or overwrite the journal.
        prior = self.committed, self.pending, self.pending_aux
        self._consume(row)
        row["sha256"] = digest(row)
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())
        except BaseException:
            self.committed, self.pending, self.pending_aux = prior
            raise
        self.rows.append(row)
        return row

    def begin(self, step): return self.append("ATTEMPT_BEGIN", step=step)
    def commit(self, step, checkpoint_sha256): return self.append("UPDATE_COMMIT", step=step, checkpoint_sha256=checkpoint_sha256)
