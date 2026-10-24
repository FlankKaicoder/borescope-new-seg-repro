"""Persistent, process-safe budget hard gates for Exp13 Phase 1-A-R2."""
from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - target audit host is Linux.
    fcntl = None

STATES = {"RESERVED", "STARTED", "SUCCEEDED", "FAILED", "UNKNOWN", "CANCELLED"}
TERMINAL = {"SUCCEEDED", "FAILED", "UNKNOWN"}
OCCUPYING = STATES - {"CANCELLED"}
STAGED = {"smoke", "pilot", "revision"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LedgerIntegrityError(RuntimeError):
    """Raised when a ledger cannot be trusted; callers must HOLD."""


@dataclass
class BudgetProtocol:
    smoke_cap: int = 4
    smoke_pilot_revision_cap: int = 120
    all_local_cap: int = 300
    max_production_sources: int = 100
    max_retry_sources: int = 10
    gpt_fallback_max: int = 14
    ledger_path: Path | None = None
    planned_calls: int = 0
    actual_calls: int = 0
    retry_calls: int = 0
    gpt_fallback_calls: int = 0
    smoke_reserved: int = 0
    history: list[dict] = field(default_factory=list)
    calls: dict[str, dict] = field(default_factory=dict)
    production_sources: set[str] = field(default_factory=set)
    retry_sources: set[str] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ledger_path is not None:
            self.ledger_path = Path(self.ledger_path)
            if self.ledger_path.exists():
                with self._critical(persist=False):
                    pass

    @property
    def _lock_path(self) -> Path:
        assert self.ledger_path is not None
        return self.ledger_path.with_name(self.ledger_path.name + ".lock")

    @contextmanager
    def _critical(self, *, persist: bool = True) -> Iterator[None]:
        """Hold thread and process locks while loading/checking/updating/writing."""
        with self._lock:
            if self.ledger_path is None:
                yield
                return
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+b") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    if self.ledger_path.exists():
                        self._load_unlocked()
                    yield
                    if persist:
                        self._sync_counters_unlocked()
                        self._save_unlocked()
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_unlocked(self) -> None:
        assert self.ledger_path is not None
        if self.ledger_path.with_name(self.ledger_path.name + ".tmp").exists():
            raise LedgerIntegrityError("HOLD: temporary or partial ledger exists")
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise LedgerIntegrityError(f"HOLD: ledger decode failed: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 2:
            raise LedgerIntegrityError("HOLD: unsupported or incomplete ledger schema")
        calls = data.get("calls")
        history = data.get("history")
        if not isinstance(calls, dict) or not isinstance(history, list):
            raise LedgerIntegrityError("HOLD: calls/history are not complete collections")
        for call_id, rec in calls.items():
            if not isinstance(rec, dict) or rec.get("call_id") != call_id:
                raise LedgerIntegrityError("HOLD: call identity is corrupt")
            if rec.get("status") not in STATES:
                raise LedgerIntegrityError("HOLD: unknown call state")
            for key in ("source_id", "roi_id", "stage", "strength", "model_config_id", "retry"):
                if key not in rec:
                    raise LedgerIntegrityError(f"HOLD: missing call field {key}")
        self.calls = copy.deepcopy(calls)
        self.history = copy.deepcopy(history)
        self._sync_counters_unlocked()

    def _sync_counters_unlocked(self) -> None:
        active = [r for r in self.calls.values() if r.get("status") in OCCUPYING]
        self.planned_calls = len(active)
        self.actual_calls = sum(r.get("status") in ({"STARTED"} | TERMINAL) for r in active)
        self.retry_calls = sum(bool(r.get("retry")) for r in active)
        self.smoke_reserved = sum(r.get("stage") == "smoke" for r in active)
        self.production_sources = {str(r["source_id"]) for r in active}
        self.retry_sources = {str(r["source_id"]) for r in active if r.get("retry")}

    def _snapshot_unlocked(self) -> dict:
        self._sync_counters_unlocked()
        occupied = self.planned_calls
        staged = sum(r.get("stage") in STAGED for r in self.calls.values() if r.get("status") in OCCUPYING)
        return {
            "schema_version": 2,
            "definition": "one LOCAL_MODEL_CALL = one ROI, one strength, one actual inference; every non-cancelled reservation occupies budget",
            "planned_calls": self.planned_calls,
            "actual_calls": self.actual_calls,
            "occupied_calls": occupied,
            "retry_calls": self.retry_calls,
            "gpt_fallback_calls": self.gpt_fallback_calls,
            "smoke_reserved": self.smoke_reserved,
            "staged_reserved": staged,
            "caps": {"smoke": self.smoke_cap, "smoke_pilot_revision": self.smoke_pilot_revision_cap, "all_local": self.all_local_cap, "unique_sources": self.max_production_sources, "retry_sources": self.max_retry_sources, "gpt_fallback": self.gpt_fallback_max},
            "production_sources": sorted(self.production_sources),
            "retry_sources": sorted(self.retry_sources),
            "calls": copy.deepcopy(self.calls),
            "history": copy.deepcopy(self.history),
            "remaining": {"all_local": self.all_local_cap - occupied, "smoke_pilot_revision": self.smoke_pilot_revision_cap - staged, "smoke": self.smoke_cap - self.smoke_reserved, "unique_sources": self.max_production_sources - len(self.production_sources), "retry_sources": self.max_retry_sources - len(self.retry_sources)},
            "execution_blocked_if_over_cap": True,
        }

    def _save_unlocked(self) -> None:
        assert self.ledger_path is not None
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_name(self.ledger_path.name + ".tmp")
        payload = json.dumps(self._snapshot_unlocked(), ensure_ascii=False, indent=2) + "\n"
        with tmp.open("w", encoding="utf-8") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.ledger_path)
        try:
            dir_fd = os.open(str(self.ledger_path.parent), os.O_DIRECTORY)
            try: os.fsync(dir_fd)
            finally: os.close(dir_fd)
        except (AttributeError, OSError):
            pass

    def _existing_signature(self, rec: dict) -> tuple:
        return tuple(rec.get(k) for k in ("source_id", "roi_id", "stage", "strength", "model_config_id", "retry"))

    def _requested_signature(self, *, source_id: str, roi_id: str, stage: str, strength: float, model_config_id: str, retry: bool) -> tuple:
        return (source_id, roi_id, stage, strength, model_config_id, bool(retry))

    def _check_existing(self, rec: dict, signature: tuple) -> dict:
        if self._existing_signature(rec) != signature:
            raise RuntimeError("call ID already exists with different source/ROI/config")
        if rec.get("status") == "CANCELLED":
            raise RuntimeError("cancelled call ID cannot be reused")
        return copy.deepcopy(rec)

    def _check_capacity_unlocked(self, *, stage: str, count: int, source_id: str, retry: bool) -> None:
        if stage not in {"smoke", "pilot", "revision", "production"}:
            raise ValueError(f"unknown stage: {stage}")
        active = [r for r in self.calls.values() if r.get("status") in OCCUPYING]
        if len(active) + count > self.all_local_cap:
            raise RuntimeError("LOCAL_MODEL_CALL budget exceeded by reservation")
        staged = sum(r.get("stage") in STAGED for r in active)
        if stage in STAGED and staged + count > self.smoke_pilot_revision_cap:
            raise RuntimeError("smoke/pilot/revision reservation budget exceeded")
        smoke = sum(r.get("stage") == "smoke" for r in active)
        if stage == "smoke" and smoke + count > self.smoke_cap:
            raise RuntimeError("smoke reservation budget exceeded")
        sources = {str(r["source_id"]) for r in active}
        if len(sources | {str(source_id)}) > self.max_production_sources:
            raise RuntimeError("unique source cap exceeded")
        retry_sources = {str(r["source_id"]) for r in active if r.get("retry")}
        if retry and str(source_id) in retry_sources:
            raise RuntimeError("retry source already has a retry call")
        if len(retry_sources | ({str(source_id)} if retry else set())) > self.max_retry_sources:
            raise RuntimeError("retry source cap exceeded")

    def _new_record(self, *, call_id: str, source_id: str, roi_id: str, stage: str, strength: float, model_config_id: str, retry: bool) -> dict:
        rec = {"call_id": call_id, "source_id": source_id, "roi_id": roi_id, "stage": stage, "strength": strength, "model_config_id": model_config_id, "retry": bool(retry), "status": "RESERVED", "real_model_call": False, "reserved_at": _now()}
        self.calls[call_id] = rec
        self.history.append({"event": "RESERVED", "call_id": call_id, "status": "RESERVED", "occupied_calls_after": len([r for r in self.calls.values() if r.get("status") in OCCUPYING])})
        return rec

    def reserve_call(self, *, call_id: str, source_id: str, roi_id: str, stage: str, strength: float, model_config_id: str = "unresolved", retry: bool = False) -> dict:
        with self._critical():
            signature = self._requested_signature(source_id=str(source_id), roi_id=str(roi_id), stage=str(stage), strength=float(strength), model_config_id=str(model_config_id), retry=retry)
            if call_id in self.calls: return self._check_existing(self.calls[call_id], signature)
            self._check_capacity_unlocked(stage=stage, count=1, source_id=str(source_id), retry=retry)
            return copy.deepcopy(self._new_record(call_id=str(call_id), source_id=str(source_id), roi_id=str(roi_id), stage=str(stage), strength=float(strength), model_config_id=str(model_config_id), retry=retry))

    def start_call(self, call_id: str, *, real_model_call: bool = False) -> dict:
        with self._critical():
            rec = self.calls.get(call_id)
            if rec is None: raise KeyError(call_id)
            if rec["status"] == "RESERVED":
                rec["status"] = "STARTED"; rec["real_model_call"] = bool(real_model_call); rec["started_at"] = _now(); self.history.append({"event": "STARTED", "call_id": call_id, "status": "STARTED"})
            elif rec["status"] != "STARTED": raise RuntimeError(f"{rec['status']} call cannot be started")
            return copy.deepcopy(rec)

    def finish_call(self, call_id: str, *, success: bool, unknown: bool = False, error: str | None = None) -> dict:
        with self._critical():
            rec = self.calls.get(call_id)
            if rec is None: raise KeyError(call_id)
            desired = "UNKNOWN" if unknown else ("SUCCEEDED" if success else "FAILED")
            if rec["status"] in TERMINAL:
                if rec["status"] != desired: raise RuntimeError("terminal call cannot change outcome")
                return copy.deepcopy(rec)
            if rec["status"] != "STARTED": raise RuntimeError("only STARTED calls may become terminal")
            rec["status"] = desired
            if error: rec["error"] = str(error)
            rec["finished_at"] = _now(); self.history.append({"event": desired, "call_id": call_id, "status": desired, "error": error})
            return copy.deepcopy(rec)

    def cancel_call(self, call_id: str, *, reason: str) -> dict:
        if not reason: raise ValueError("cancellation requires an audit reason")
        with self._critical():
            rec = self.calls.get(call_id)
            if rec is None: raise KeyError(call_id)
            if rec["status"] != "RESERVED": raise RuntimeError("only RESERVED calls may be cancelled")
            rec["status"] = "CANCELLED"; rec["cancelled_at"] = _now(); rec["cancel_reason"] = str(reason); self.history.append({"event": "CANCELLED", "call_id": call_id, "status": "CANCELLED", "reason": str(reason)})
            return copy.deepcopy(rec)

    def reserve_source_group(self, *, source_id: str, roi_ids: list[str], stage: str, strength: float, model_config_id: str = "unresolved", retry: bool = False) -> list[dict]:
        if not roi_ids: return []
        with self._critical():
            ids = [f"{source_id}:{roi}:s{strength}" for roi in roi_ids]
            signatures = [self._requested_signature(source_id=str(source_id), roi_id=str(roi), stage=str(stage), strength=float(strength), model_config_id=str(model_config_id), retry=retry) for roi in roi_ids]
            missing: list[tuple[str, str]] = []; existing: list[dict] = []
            for call_id, roi, signature in zip(ids, roi_ids, signatures):
                if call_id in self.calls: existing.append(self._check_existing(self.calls[call_id], signature))
                else: missing.append((call_id, str(roi)))
            if missing:
                self._check_capacity_unlocked(stage=stage, count=len(missing), source_id=str(source_id), retry=retry)
                for call_id, roi in missing: self._new_record(call_id=call_id, source_id=str(source_id), roi_id=roi, stage=str(stage), strength=float(strength), model_config_id=str(model_config_id), retry=retry)
            return existing + [copy.deepcopy(self.calls[cid]) for cid, _ in missing]

    def reserve(self, count: int, *, kind: str, source_id: str | None = None, strength: float | None = None) -> None:
        if count < 0: raise ValueError("count must be non-negative")
        stage = "smoke" if "smoke" in kind else ("pilot" if "pilot" in kind else ("production" if "production" in kind else "revision"))
        sid = str(source_id) if source_id is not None else f"legacy-source-{uuid.uuid4()}"
        for _ in range(count): self.reserve_call(call_id=f"legacy-{uuid.uuid4()}", source_id=sid, roi_id="legacy", stage=stage, strength=float(strength if strength is not None else 0.0))

    def record_actual(self, *, kind: str, source_id: str, strength: float, retry: bool = False) -> None:
        stage = "smoke" if "smoke" in kind else ("production" if "production" in kind else "pilot"); call_id = str(uuid.uuid4())
        self.reserve_call(call_id=call_id, source_id=source_id, roi_id="legacy", stage=stage, strength=strength, retry=retry); self.start_call(call_id); self.finish_call(call_id, success=True)

    def record_gpt_fallback(self, *, source_id: str) -> None:
        raise RuntimeError("GPT fallback is unauthorized")

    def snapshot(self) -> dict:
        with self._critical(persist=False): return self._snapshot_unlocked()
