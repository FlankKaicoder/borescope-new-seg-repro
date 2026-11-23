"""Prepared formal entry: refuses absent independent Gate before model imports.

G1 runs only synthetic CPU tests of this control flow. A checkpoint's atomic
pointer and the fsync'd journal must both agree before an update can be resumed.
An ambiguous physical attempt is preserved as a hard Gate, never replayed.
"""
import argparse
import copy
import json
import traceback
from pathlib import Path

from tools.exp17.formal.authorization import (HardGate, RunJournal, RunLease, claim_fresh, digest,
    exclusive_json, file_sha, load, protocol_bindings, require, validate_authorization, verify_existing_claim)


def _rows(path, committed):
    if not path.exists():
        require(committed == 0, "Committed update metrics are missing")
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    require(all(line.endswith("\n") for line in lines), "Truncated metrics ledger")
    rows = [json.loads(line) for line in lines]
    require([r["step"] for r in rows] == list(range(1, committed+1)), "Metrics ledger differs from committed updates")
    return rows


def _epoch_callbacks(adapter, epoch, payload, output, best):
    from tools.exp17.formal.checkpoint import atomic_json
    from tools.exp17.formal.exports import save_best_bundle
    from tools.exp17.formal.validation import best_update
    from tools.exp14.formal_e0_e1 import hash_state
    if epoch % 5:
        return best
    reports = adapter.gradient_diagnostics()
    atomic_json(output / f"diagnostics_epoch{epoch:03d}.json", reports)
    result = copy.deepcopy(best)
    for role in ("master", "ema"):
        metrics = adapter.evaluate(epoch, role)
        atomic_json(output / f"VAL_epoch{epoch:03d}_{role}.json", metrics)
        canonical = {field:hash_state({k:v for k,v in payload[field].items()
                                      if not k.startswith("region_projection.")}) for field in ("state","ema")}
        record, selected = best_update(result.get(role), epoch=epoch, role=role,
            master_sha256=canonical["state"], ema_sha256=canonical["ema"], evaluation=metrics)
        if selected:
            receipt = save_best_bundle(output / "best", payload, record)
            record["checkpoint"] = receipt
        result[role] = record
    return result


def run_session(adapter, protocol, bindings, output, journal, store, *, recovered=None, synthetic_stop_after=None,
                legacy_cache_bindings=None):
    """Same checkpoint/ledger/epoch logic for native adapter and CPU toy tests."""
    from tools.exp17.formal.checkpoint import capture_training_state, restore_training_state, atomic_json
    from tools.exp17.formal.monitor import append_json, ensure_space, epoch_summary
    from tools.exp17.formal.sampler import SealedEpochCache
    output = Path(output); trainer = adapter.trainer
    require(not any(r["event"] == "TRAIN_COMPLETE" for r in journal.rows), "Completed experiment cannot restart")
    require(synthetic_stop_after is None or getattr(adapter, "synthetic_only", False), "Real formal run cannot request a truncated budget")
    require(protocol["training"]["epochs"] == 100 and protocol["effective_updates"] == 2100,
            "Formal epoch/update budget changed")
    all_rows = _rows(output / "steps.jsonl", journal.committed)
    best = {}; cursor = None; begin_epoch = 0; extra = {}
    if recovered is not None:
        position, sampler, extra = restore_training_state(trainer, recovered, bindings=bindings)
        require(position["global_update"] == journal.committed, "Checkpoint and permanent journal disagree")
        begin_epoch = position["epoch"]; best = extra.get("best", {})
        trainer.epoch = begin_epoch
        if position["batch_position"] == 21 and extra.get("epoch_events_complete"):
            begin_epoch += 1
        elif position["batch_position"]:
            cache = SealedEpochCache.open(sampler["cache_directory"], bindings=bindings,
                expected_manifest_sha256=sampler["cache_manifest_sha256"])
            cursor = cache.cursor(); cursor.load_state_dict(sampler); adapter.set_sampler(cursor)
    else:
        require(journal.committed == 0, "Fresh model cannot enter an already-started update ledger")
        initial = capture_training_state(trainer, bindings=bindings,
            position={"epoch":0,"batch_position":0,"global_update":0},
            sampler_state={"schema":"BEFORE_FIRST_EPOCH","epoch":0,"next_batch":0},
            extra={"best":{},"epoch_events_complete":False})
        initial_receipt = store.save_latest(initial)
        journal.append("INITIAL_STATE_COMMIT", checkpoint_sha256=initial_receipt["sha256"])
    completed = journal.committed
    for epoch in range(begin_epoch, 100):
        if cursor is None:
            ensure_space(output, required_bytes=protocol["resources"]["before_epoch_min_free_bytes"])
            adapter.start_epoch(epoch)
            cache = adapter.build_cache(epoch, output / "epoch_cache" / f"epoch_{epoch:03d}")
            cache_bytes = sum(r["bytes"] for r in cache.manifest["batches"])
            require(cache_bytes <= protocol["resources"].get("estimated_max_epoch_cache_bytes", cache_bytes),
                    "Sealed epoch cache exceeds the frozen resource estimate")
            cursor = cache.cursor(); adapter.set_sampler(cursor)
        require(cursor.state_dict()["epoch"] == epoch, "Restored sampler belongs to another epoch")
        while cursor.next_index < 21:
            step = epoch*21 + cursor.next_index + 1
            require(step == journal.committed + 1, "Native batch cursor skipped or repeated an update")
            ensure_space(output, required_bytes=protocol["resources"]["before_checkpoint_min_free_bytes"])
            batch = cursor.next_batch()
            journal.begin(step)
            row = adapter.train_batch(batch, step)
            require(row["step"] == row["effective_updates"] == row["optimizer_attempts"] == step,
                    "Actual optimizer update count differs from its permanent reservation")
            append_json(output / "steps.jsonl", row)
            all_rows.append(row)
            extra = {"best": best, "epoch_events_complete": False}
            payload = capture_training_state(trainer, bindings=bindings,
                position={"epoch": epoch, "batch_position": cursor.next_index, "global_update": step},
                sampler_state=cursor.state_dict(), extra=extra)
            receipt = store.save_latest(payload)
            require(receipt["bytes"] <= protocol["resources"].get("estimated_max_recovery_bytes", receipt["bytes"]),
                    "Recovery checkpoint exceeds the frozen resource estimate")
            journal.commit(step, receipt["sha256"])
            completed = step
            # Retire only caches produced by this new run, after durable update
            # commit, retaining current+previous. Histories/assets are untouched.
            old = output / "epoch_cache" / f"epoch_{epoch-2:03d}"
            if epoch >= 2 and old.is_dir():
                old_manifest = old / "manifest.json"
                manifest_sha = file_sha(old_manifest)
                cache_bindings = bindings
                if legacy_cache_bindings is not None:
                    cache_bindings = legacy_cache_bindings(old, manifest_sha)
                SealedEpochCache.open(old, bindings=cache_bindings, expected_manifest_sha256=manifest_sha).retire(
                    owned_cache_root=output / "epoch_cache", committed_sampler_state=cursor.state_dict(), retention_authorized=True)
            if synthetic_stop_after == step:
                return {"status":"SYNTHETIC_PAUSE_AT_COMMITTED_UPDATE", "updates":step}
            del batch, payload
        journal.append("AUX_BEGIN", epoch=epoch+1)
        epoch_rows = all_rows[epoch*21:(epoch+1)*21]
        report = epoch_summary(epoch+1, epoch_rows, ema_updates=trainer.ema.updates)
        atomic_json(output / f"epoch{epoch+1:03d}.json", report)
        payload, _ = store.load_latest()
        best = _epoch_callbacks(adapter, epoch+1, payload, output, best)
        final_extra = {"best":best, "epoch_events_complete":True}
        payload = capture_training_state(trainer, bindings=bindings,
            position={"epoch":epoch, "batch_position":21, "global_update":completed},
            sampler_state=cursor.state_dict(), extra=final_extra)
        receipt = store.save_latest(payload, allow_same_update=True)
        if (epoch+1) % 5 == 0:
            store.save_artifact(f"epoch{epoch+1:03d}_full_recovery.pt", payload, role="periodic")
        journal.append("AUX_COMMIT", epoch=epoch+1, checkpoint_sha256=receipt["sha256"])
        cursor = None
    require(completed == 2100 and trainer.ema.updates == 2100, "Final formal update budget incomplete")
    from tools.exp17.formal.exports import write_final_native
    endpoints = {role:write_final_native(trainer, output / "final", bindings, role=role)
                 for role in ("master","ema")}
    atomic_json(output / "final_endpoints.json", endpoints)
    journal.append("TRAIN_COMPLETE", updates=2100, endpoints=endpoints)
    return {"status":"FORMAL_100_EPOCH_COMPLETE_PENDING_INDEPENDENT_RESULT_GATE", "updates":2100,
            "best":best,"endpoints":endpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-authorization", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    protocol, authorization = validate_authorization(args.protocol,args.authorization,args.output,repository)
    bindings = protocol_bindings(protocol,file_sha(args.protocol))
    config = load(args.runtime_config)
    require(authorization.get("runtime_config_sha256") == file_sha(args.runtime_config), "Private runtime configuration differs")
    require(protocol["resources"]["capacity_status"] == "PASS_ESTIMATED_CAPACITY", "Resource budget is on HOLD")
    recovered = None
    if args.resume_authorization:
        anchor = verify_existing_claim(authorization,args.output)
    else:
        anchor = claim_fresh(authorization,args.output)
    with RunLease(anchor):
        journal = RunJournal(anchor / "updates.jsonl", fresh=not bool(args.resume_authorization))
        from tools.exp17.formal.checkpoint import AtomicCheckpointStore
        store = AtomicCheckpointStore(args.output / "recovery",bindings)
        if args.resume_authorization:
            recovered, receipt = store.load_latest()
            last_sha = next((r["checkpoint_sha256"] for r in reversed(journal.rows)
                             if r["event"] in ("INITIAL_STATE_COMMIT","UPDATE_COMMIT","AUX_COMMIT")), None)
            require(receipt["sha256"] == last_sha, "Latest checkpoint differs from the permanent committed journal")
            gate = load(args.resume_authorization)
            require(gate == {"stage":"EXP17_G1_RESUME_AUTHORIZATION", "gate":"CHATGPT_RECOVERY_GATE_PASS",
                "authorization_sha256":digest(authorization),"checkpoint_sha256":receipt["sha256"],
                "journal_sha256":file_sha(anchor / "updates.jsonl"),"committed_update":journal.committed,
                "output":str(args.output.resolve())}, "Separate exact-state recovery Gate required")
            require(journal.committed <= 2100 and not any(r["event"] == "TRAIN_COMPLETE" for r in journal.rows),
                    "Completed experiment cannot restart")
        # The only heavy/runtime import is below all independent authorization,
        # source/protocol/output/budget/claim and recovery checks.
        from tools.exp17.formal.native_adapter import NativeAdapter
        adapter = None
        try:
            adapter = NativeAdapter(config,protocol,args.output,bindings)
            result = run_session(adapter,protocol,bindings,args.output,journal,store,recovered=recovered)
            exclusive_json(args.output / "completion.json",result)
        except BaseException as error:
            import sys
            evidence = {"status":"EXP17_FORMAL_HARD_GATE","error_type":type(error).__name__,
                        "message":str(error),"traceback":traceback.format_exc(),"updates":journal.committed}
            try:
                journal.append("HARD_GATE",error_type=type(error).__name__,message=str(error))
                failure = args.output / f"hard_gate_{len(journal.rows):05d}.json"
                exclusive_json(failure,evidence)
            except BaseException as save_error:
                # Disk failure must not replace the original scientific error.
                print(json.dumps({**evidence,"evidence_save_error":str(save_error)}),file=sys.stderr,flush=True)
            raise
        finally:
            if adapter is not None:adapter.trainer.close()


if __name__=="__main__":main()
