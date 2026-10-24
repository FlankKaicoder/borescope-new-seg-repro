"""Budget and provenance protocol; no model execution is performed here."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetProtocol:
    smoke_cap: int = 4
    smoke_pilot_revision_cap: int = 120
    all_local_cap: int = 300
    max_production_sources: int = 100
    max_retry_sources: int = 10
    gpt_fallback_max: int = 14
    planned_calls: int = 0
    actual_calls: int = 0
    retry_calls: int = 0
    gpt_fallback_calls: int = 0
    history: list[dict] = field(default_factory=list)

    def reserve(self, count: int, *, kind: str, source_id: str | None = None, strength: float | None = None) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        projected = self.actual_calls + count
        if projected > self.all_local_cap:
            raise RuntimeError(f"LOCAL_MODEL_CALL budget exceeded: {projected}>{self.all_local_cap}")
        if kind in {"smoke_planned", "pilot_planned", "revision_reserve"} and self.planned_calls + count > self.smoke_pilot_revision_cap:
            raise RuntimeError(f"smoke/pilot/revision budget exceeded: {self.planned_calls + count}>{self.smoke_pilot_revision_cap}")
        self.planned_calls += count
        self.history.append({"event": "planned", "count": count, "kind": kind, "source_id": source_id, "strength": strength})

    def record_actual(self, *, kind: str, source_id: str, strength: float, retry: bool = False) -> None:
        projected = self.actual_calls + 1
        if projected > self.all_local_cap:
            raise RuntimeError(f"LOCAL_MODEL_CALL budget exceeded: {projected}>{self.all_local_cap}")
        if retry and self.retry_calls + 1 > self.max_retry_sources:
            raise RuntimeError(f"retry budget exceeded: {self.retry_calls + 1}>{self.max_retry_sources}")
        self.actual_calls = projected
        self.retry_calls += int(retry)
        self.history.append({"event": "actual", "count": 1, "kind": kind, "source_id": source_id, "strength": strength, "retry": retry})

    def record_gpt_fallback(self, *, source_id: str) -> None:
        if self.gpt_fallback_calls + 1 > self.gpt_fallback_max:
            raise RuntimeError("GPT fallback budget exceeded; separate authorization required")
        self.gpt_fallback_calls += 1
        self.history.append({"event": "gpt_fallback", "count": 1, "source_id": source_id, "authorization": "REQUIRED"})

    def snapshot(self) -> dict:
        return {
            "definition": "one LOCAL_MODEL_CALL = one ROI, one strength, one actual inference",
            "planned_calls": self.planned_calls,
            "actual_calls": self.actual_calls,
            "retry_calls": self.retry_calls,
            "gpt_fallback_calls": self.gpt_fallback_calls,
            "caps": {"smoke": self.smoke_cap, "smoke_pilot_revision": self.smoke_pilot_revision_cap, "all_local": self.all_local_cap, "production_sources": self.max_production_sources, "retry_sources": self.max_retry_sources, "gpt_fallback": self.gpt_fallback_max},
            "execution_blocked_if_over_cap": True,
            "history": self.history,
        }
