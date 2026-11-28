"""Protocol-driven joint-update engine and independently gated native boundary."""
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

from .optimizer import joint_gradient_stats, validate_gradients
from .checkpoint import state_hash


class SyntheticExecutor:
    def __init__(self, model, optimizer, scheduler, ema, *, head_active_updates=0, global_update=0):
        if getattr(model, "exp18_synthetic", False) is not True or any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("Current executor accepts explicitly synthetic CPU models only")
        self.model, self.optimizer, self.scheduler, self.ema = model, optimizer, scheduler, ema
        self.global_update, self.head_active_updates = global_update, head_active_updates

    def step(self, loss, *, head_active):
        if type(head_active) is not bool:
            raise ValueError("Explicit matching-derived head activity required")
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError("Step entered with uncleared gradients")
        loss.backward()
        validate_gradients(self.model, head_active=head_active)
        before_clip = joint_gradient_stats(self.model)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite clipped gradient norm")
        after_clip = joint_gradient_stats(self.model)
        self.ema.set_head_active(head_active)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model)
        self.global_update += 1
        self.head_active_updates += int(head_active)
        if self.ema.updates != self.global_update or self.ema.head_active_updates != self.head_active_updates:
            raise ValueError("Global/head clocks disagree after update")
        return {"global_update": self.global_update, "head_active_updates": self.head_active_updates,
                "head_active": head_active, "gradient_norm": float(norm),
                "gradient_before_clip": before_clip, "gradient_after_clip": after_clip,
                "clip_coefficient": min(1., 10. / (float(norm) + 1e-6))}


class AppendOnlyJournal:
    """A hash-chain reservation/commit journal; ambiguous attempts never resume."""
    def __init__(self, path, bindings, *, create=False):
        self.path, self.bindings = Path(path), copy.deepcopy(bindings)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8") as stream:
                stream.flush(); os.fsync(stream.fileno())
        self.rows, self.committed, self.pending, self.pending_aux = [], 0, None, None
        previous = "0" * 64
        serialized = self.path.read_bytes()
        self._verified_bytes = len(serialized)
        for line in serialized.decode("utf-8").splitlines(keepends=True):
            if not line.endswith("\n"):
                raise ValueError("Truncated journal")
            row = json.loads(line)
            stored = row.pop("sha256")
            if (row["previous"] != previous or row["sequence"] != len(self.rows)
                    or row["bindings"] != self.bindings or self._hash(row) != stored):
                raise ValueError("Journal chain/binding changed")
            row["sha256"] = stored
            self._consume(row)
            self.rows.append(row)
            previous = stored
        if self.pending is not None or self.pending_aux is not None:
            raise ValueError("Ambiguous interrupted update/evaluation: independent recovery required")

    @staticmethod
    def _hash(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def _consume(self, row):
        event = row["event"]
        if event == "ATTEMPT_BEGIN":
            if self.pending is not None or self.pending_aux is not None or row["step"] != self.committed + 1:
                raise ValueError("Journal update repeated/skipped")
            self.pending = row["step"]
        elif event == "UPDATE_COMMIT":
            if self.pending != row["step"]:
                raise ValueError("Commit has no matching reservation")
            self.committed, self.pending = row["step"], None
        elif event == "AUX_BEGIN":
            if self.pending is not None or self.pending_aux is not None:
                raise ValueError("Concurrent auxiliary event")
            self.pending_aux = row["epoch"]
        elif event == "AUX_COMMIT":
            if self.pending_aux != row["epoch"]:
                raise ValueError("Auxiliary commit has no reservation")
            self.pending_aux = None

    def append(self, event, **fields):
        # The journal may have been opened before this process acquired its
        # native lease. Reject a stale in-memory tail BEFORE reserving an update.
        if self.path.stat().st_size != self._verified_bytes:
            raise ValueError("Stale journal writer: permanent tail changed; reopen complete recovery")
        row = {"sequence": len(self.rows), "previous": self.rows[-1]["sha256"] if self.rows else "0" * 64,
               "bindings": self.bindings, "event": event, **fields}
        row["sha256"] = self._hash(row)
        self._consume(row)
        encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush(); os.fsync(stream.fileno())
        self._verified_bytes += len(encoded)
        self.rows.append(row)
        return row


class ExperimentExecutor:
    """Drive an injected backend, including cache, recovery, VAL and final callbacks.

    Adapter contract: model/optimizer/scheduler/ema/scaler attributes;
    start_epoch(epoch); build_cache(epoch)->cursor; open_cache(sampler)->cursor;
    forward_batch(batch,step)->JointForward; before_batch(step); capture(...);
    restore(record)->position/sampler/counters/extra; evaluate(epoch,role);
    export(role); optionally diagnostics(epoch), after_batch(row).

    Every real backend is rejected by the current CPU protocol BEFORE an adapter
    factory can construct a model. Synthetic mode uses the same event machine.
    """
    def __init__(self, adapter, protocol, bindings, store, journal, *, synthetic=False):
        self.protocol, self.bindings = copy.deepcopy(protocol), copy.deepcopy(bindings)
        if not synthetic:
            from .protocol import require_operation
            require_operation("train", protocol)
        elif getattr(adapter.model, "exp18_synthetic", False) is not True:
            raise ValueError("Synthetic event executor requires an explicitly synthetic model")
        self.adapter, self.store, self.journal, self.synthetic = adapter, store, journal, synthetic
        self.epochs = protocol["train"]["epochs"]
        self.batches = protocol["train"]["batches_per_epoch"]
        if self.epochs not in (100, 200) or self.batches != 21:
            raise ValueError("Only independent H100/H200 21-batch protocols supported")
        self.global_update = journal.committed
        self.head_active_updates = 0

    def _capture(self, epoch, cursor, *, events_complete=False, revision=0):
        sampler = cursor.state_dict() if cursor is not None else {"epoch": epoch, "next_batch": 0}
        position = {"epoch": epoch, "batch_position": sampler["next_batch"],
                    "global_update": self.global_update, "revision": revision}
        return self.adapter.capture(position=position, sampler_state=sampler,
            head_active_updates=self.head_active_updates,
            extra={"epoch_events_complete": events_complete})

    def _update(self, batch, step, *, observe=None):
        adapter = self.adapter
        adapter.before_batch(step)
        adapter.optimizer.zero_grad(set_to_none=True)
        result = adapter.forward_batch(batch, step)
        if not bool(torch.isfinite(result.total)):
            raise ValueError("Nonfinite forward loss")
        scaler = getattr(adapter, "scaler", None)
        if scaler is not None:
            scaler.scale(result.total).backward()
            scaler.unscale_(adapter.optimizer)
        else:
            result.total.backward()
        validate_gradients(adapter.model, head_active=result.head_active)
        before_clip = joint_gradient_stats(adapter.model)
        diagnostics = observe(result) if observe is not None else None
        norm = torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), self.protocol["train"]["clip_global_norm"])
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite joint gradient norm")
        after_clip = joint_gradient_stats(adapter.model)
        observed = []
        hook = adapter.optimizer.register_step_pre_hook(lambda *_: observed.append(True))
        adapter.ema.set_head_active(result.head_active)
        try:
            if scaler is not None:
                scaler.step(adapter.optimizer); scaler.update()
            else:
                adapter.optimizer.step()
        finally:
            hook.remove()
        if len(observed) != 1:
            raise ValueError("AMP skipped or repeated an optimizer update")
        adapter.optimizer.zero_grad(set_to_none=True)
        adapter.ema.update(adapter.model)
        self.global_update += 1
        self.head_active_updates += int(result.head_active)
        if adapter.ema.updates != step or adapter.ema.head_active_updates != self.head_active_updates:
            raise ValueError("Optimizer/EMA active clocks disagree")
        row = {"global_update": step, "head_active_updates": self.head_active_updates,
               "head_active": result.head_active, "matching_sha256": result.matching_sha256,
               "loss": float(result.total.detach()), "state_loss": float(result.state.detach()),
               "gradient_norm": float(norm), "gradient_before_clip": before_clip,
               "gradient_after_clip": after_clip,
               "clip_coefficient": min(1., self.protocol["train"]["clip_global_norm"] / (float(norm) + 1e-6))}
        if hasattr(result, "yolo") and hasattr(result, "supcon"):
            row["yolo_loss"] = float(result.yolo.detach())
            row["supcon_loss"] = float(result.supcon.detach())
        elif not self.synthetic:
            raise ValueError("Native joint forward omitted YOLO/SupCon components")
        if diagnostics is not None:
            row.update(diagnostics)
        if hasattr(adapter, "after_batch"):
            adapter.after_batch(row)
        return row

    def run(self, *, recovered=None, stop_after=None):
        if stop_after is not None and not self.synthetic:
            raise ValueError("Native formal budget cannot be truncated")
        if any(row["event"] == "TRAIN_COMPLETE" for row in self.journal.rows):
            raise ValueError("Completed experiment cannot restart")
        cursor, first_epoch = None, 0
        if recovered is not None:
            committed_states = [row for row in self.journal.rows if "recovery_digest" in row]
            if not committed_states or state_hash(recovered) != committed_states[-1]["recovery_digest"]:
                raise ValueError("Recovery is not the latest journal-bound complete state")
            state = self.adapter.restore(recovered)
            position = state["position"]
            if position["global_update"] != self.journal.committed:
                raise ValueError("Recovery and permanent journal disagree")
            self.global_update = position["global_update"]
            self.head_active_updates = state["head_active_updates"]
            first_epoch = position["epoch"]
            if position["batch_position"] == self.batches and state["extra"].get("epoch_events_complete"):
                first_epoch += 1
            elif position["batch_position"]:
                cursor = self.adapter.open_cache(state["sampler"])
        elif self.journal.rows:
            raise ValueError("Started journal requires complete recovery")
        else:
            record = self._capture(0, None)
            receipt = self.store.save(record)
            self.journal.append("INITIAL_STATE_COMMIT", checkpoint_sha256=receipt["sha256"], recovery_digest=state_hash(record))
        for epoch in range(first_epoch, self.epochs):
            if cursor is None:
                self.adapter.start_epoch(epoch)
                cursor = self.adapter.build_cache(epoch)
            while cursor.next_index < self.batches:
                step = epoch * self.batches + cursor.next_index + 1
                if step != self.journal.committed + 1:
                    raise ValueError("Sealed batch cursor repeated/skipped an update")
                batch = cursor.next_batch()
                self.journal.append("ATTEMPT_BEGIN", step=step)
                row = self._update(batch, step)
                record = self._capture(epoch, cursor)
                receipt = self.store.save(record)
                self.journal.append("UPDATE_COMMIT", step=step, checkpoint_sha256=receipt["sha256"],
                                    recovery_digest=state_hash(record), row=row)
                # Retain no previous FP32 N pack or full CPU recovery copy when
                # the next batch is materialized. Artifacts remain immutable.
                batch = record = None
                if stop_after == step:
                    return {"status": "SYNTHETIC_COMMITTED_STOP", "updates": step}
            self.journal.append("AUX_BEGIN", epoch=epoch + 1)
            if (epoch + 1) % self.protocol["future_validation"]["interval_epochs"] == 0:
                if hasattr(self.adapter, "diagnostics"):
                    self.adapter.diagnostics(epoch + 1)
                for role in self.protocol["future_validation"]["roles"]:
                    self.adapter.evaluate(epoch + 1, role)
            record = self._capture(epoch, cursor, events_complete=True, revision=1)
            receipt = self.store.save(record)
            self.journal.append("AUX_COMMIT", epoch=epoch + 1, checkpoint_sha256=receipt["sha256"],
                                recovery_digest=state_hash(record))
            cursor = None
        if self.global_update != self.epochs * self.batches:
            raise ValueError("Formal update budget incomplete")
        endpoints = {role: self.adapter.export(role) for role in self.protocol["future_validation"]["roles"]}
        self.journal.append("TRAIN_COMPLETE", updates=self.global_update)
        return {"status": "COMPLETE_PENDING_RESULT_GATE", "updates": self.global_update, "endpoints": endpoints}


def run_native(adapter_factory, protocol, bindings, store, journal, *, recovered=None):
    # Reject before the factory gets an opportunity to import/construct a model.
    from .protocol import require_operation
    require_operation("train", protocol)
    adapter = adapter_factory()
    try:
        return ExperimentExecutor(adapter, protocol, bindings, store, journal).run(recovered=recovered)
    finally:
        if hasattr(adapter, "close"):
            adapter.close()
