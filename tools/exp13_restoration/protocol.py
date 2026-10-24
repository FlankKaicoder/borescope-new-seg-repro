"""Persistent budget hard gates for Exp13 Phase 1-A-R1."""
from __future__ import annotations
import json, os, threading, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

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
        if self.ledger_path:
            self.ledger_path = Path(self.ledger_path)
            if self.ledger_path.exists():
                data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                for key in ("planned_calls", "actual_calls", "retry_calls", "gpt_fallback_calls", "smoke_reserved"):
                    setattr(self, key, int(data.get(key, 0)))
                self.history = list(data.get("history", [])); self.calls = dict(data.get("calls", {}))
                self.production_sources = set(data.get("production_sources", [])); self.retry_sources = set(data.get("retry_sources", []))

    def _save(self) -> None:
        if not self.ledger_path: return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.ledger_path)

    def _check_stage(self, stage: str, n: int) -> None:
        if stage == "smoke" and self.smoke_reserved + n > self.smoke_cap: raise RuntimeError("smoke budget exceeded")
        if stage in {"smoke", "pilot", "revision"} and self.planned_calls + n > self.smoke_pilot_revision_cap: raise RuntimeError("smoke/pilot/revision budget exceeded")
        if self.actual_calls + n > self.all_local_cap: raise RuntimeError("LOCAL_MODEL_CALL budget exceeded")

    def reserve_call(self, *, call_id: str, source_id: str, roi_id: str, stage: str, strength: float, model_config_id: str = "unresolved", retry: bool = False) -> dict:
        with self._lock:
            if call_id in self.calls: return dict(self.calls[call_id])
            self._check_stage(stage, 1)
            if stage == "production":
                if len(self.production_sources | {source_id}) > self.max_production_sources: raise RuntimeError("production source cap exceeded")
                self.production_sources.add(source_id)
            if retry:
                if len(self.retry_sources | {source_id}) > self.max_retry_sources: raise RuntimeError("retry source cap exceeded")
                self.retry_sources.add(source_id)
            rec = {"call_id": call_id, "source_id": source_id, "roi_id": roi_id, "stage": stage, "strength": strength, "model_config_id": model_config_id, "status": "RESERVED", "real_model_call": False, "retry": bool(retry), "reserved_at": _now()}
            self.calls[call_id] = rec; self.planned_calls += 1
            if stage == "smoke": self.smoke_reserved += 1
            self.history.append({"event": "RESERVED", "call_id": call_id, "remaining_budget": self.all_local_cap - self.actual_calls}); self._save(); return dict(rec)

    def start_call(self, call_id: str, *, real_model_call: bool = False) -> dict:
        with self._lock:
            rec = self.calls[call_id]
            if rec["status"] == "RESERVED": self.actual_calls += 1
            if rec["status"] in {"RESERVED", "STARTED"}:
                rec["status"] = "STARTED"; rec["real_model_call"] = bool(real_model_call); rec["started_at"] = rec.get("started_at", _now())
            self._save(); return dict(rec)

    def finish_call(self, call_id: str, *, success: bool, unknown: bool = False, error: str | None = None) -> dict:
        with self._lock:
            rec = self.calls[call_id]
            if rec["status"] not in {"SUCCEEDED", "FAILED", "UNKNOWN"}:
                rec["status"] = "UNKNOWN" if unknown else ("SUCCEEDED" if success else "FAILED")
                if rec["retry"]: self.retry_calls += 1
                if error: rec["error"] = error
                rec["finished_at"] = _now(); self.history.append({"event": rec["status"], "call_id": call_id, "remaining_budget": self.all_local_cap - self.actual_calls})
            self._save(); return dict(rec)

    def reserve_source_group(self, *, source_id: str, roi_ids: list[str], stage: str, strength: float, model_config_id: str = "unresolved", retry: bool = False) -> list[dict]:
        with self._lock:
            ids = [f"{source_id}:{roi}:s{strength}" for roi in roi_ids]; missing = [x for x in ids if x not in self.calls]
            self._check_stage(stage, len(missing))
            if stage == "production" and len(self.production_sources | {source_id}) > self.max_production_sources: raise RuntimeError("production source cap exceeded")
            if retry and len(self.retry_sources | {source_id}) > self.max_retry_sources: raise RuntimeError("retry source cap exceeded")
            return [self.reserve_call(call_id=cid, source_id=source_id, roi_id=roi, stage=stage, strength=strength, model_config_id=model_config_id, retry=retry) for cid, roi in zip(ids, roi_ids) if cid not in self.calls]

    def reserve(self, count: int, *, kind: str, source_id: str | None = None, strength: float | None = None) -> None:
        stage = "smoke" if "smoke" in kind else ("pilot" if "pilot" in kind else "revision")
        with self._lock: self._check_stage(stage, count); self.planned_calls += count; self.smoke_reserved += count if stage == "smoke" else 0; self.history.append({"event": "planned", "count": count, "kind": kind, "source_id": source_id, "strength": strength}); self._save()

    def record_actual(self, *, kind: str, source_id: str, strength: float, retry: bool = False) -> None:
        stage = "smoke" if "smoke" in kind else ("production" if "production" in kind else "pilot"); cid = str(uuid.uuid4())
        self.reserve_call(call_id=cid, source_id=source_id, roi_id="legacy", stage=stage, strength=strength, retry=retry); self.start_call(cid); self.finish_call(cid, success=True)

    def record_gpt_fallback(self, *, source_id: str) -> None: raise RuntimeError("GPT fallback is unauthorized")

    def snapshot(self) -> dict:
        return {"definition": "one LOCAL_MODEL_CALL = one ROI, one strength, one actual inference", "planned_calls": self.planned_calls, "actual_calls": self.actual_calls, "retry_calls": self.retry_calls, "gpt_fallback_calls": self.gpt_fallback_calls, "smoke_reserved": self.smoke_reserved, "caps": {"smoke": self.smoke_cap, "smoke_pilot_revision": self.smoke_pilot_revision_cap, "all_local": self.all_local_cap, "production_sources": self.max_production_sources, "retry_sources": self.max_retry_sources, "gpt_fallback": self.gpt_fallback_max}, "production_sources": sorted(self.production_sources), "retry_sources": sorted(self.retry_sources), "calls": self.calls, "history": self.history, "remaining": {"all_local": self.all_local_cap - self.actual_calls, "smoke_pilot_revision": self.smoke_pilot_revision_cap - self.planned_calls, "smoke": self.smoke_cap - self.smoke_reserved}, "execution_blocked_if_over_cap": True}
