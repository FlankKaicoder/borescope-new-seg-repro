"""Exact, once-only update105/AUX5 engineering migration; no budget reset.

The original authorization, claims, scientific protocol and checkpoint remain
unchanged. A separately accepted addendum binds the repaired production bytes.
Ordinary G1 validation and checkpoint/cache equality checks stay strict.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback

from tools.exp17.formal.authorization import (
    ACTIONS, AUTH_STAGE, EXPERIMENT, HardGate, RunJournal, RunLease,
    control_anchor, digest, exclusive_json, file_sha, load, protocol_bindings,
    require, source_catalog, verify_existing_claim,
)

SCHEMA = "EXP17_J2_R1_V1"
PROTOCOL_SHA = "613e4e154a872e0fd88791013f8e63716100470bb4a2f73084b702bb5cd37e44"
CHECKPOINT_SHA = "b8671bb8db32b40209020232478fa92e258e1d11681121d265323dc01553f83a"
JOURNAL_SHA = "3b7ef98469bb0a45587e2dd73ee7abded1a1e712871388faa34c4862cd7576d7"
CONV_SHA = "548ee1f3fa28542ad46b920e4d4271594efa09bc76c4ab3067a08a966f30648d"
LEGACY_MANIFESTS = {3: "ccf39a98a57391e4ea6c184d19277c6c4eb64bed28cd833be183ad42bee4eba4",
                    4: "9a17ccfcc1de2882606f32f3277d9cf83851d5c32da884c918c5382f68af8569"}
CHANGED_FILES = {
    "tools/exp17/engineering/model.py", "tools/exp17/formal/executor.py",
    "tools/exp17/formal/repair_r1.py", "tools/exp17/formal/repair_evidence.py",
}
R1_ACTIONS = ["repair_isolation", "accept_posthoc_master_evidence",
              "recover_epoch5_aux", "resume_remaining_seed42_budget"]
STAGES = ("MASTER_ASSOCIATION", "EMA_ROLE", "BEST_READY", "CHECKPOINT_READY")


def verify_file(record, *, prefix=False):
    path = Path(record["path"])
    require(path.is_file() and not path.is_symlink(), "Missing/symlink R1 evidence: " + str(path))
    size = record["bytes"]
    require(path.stat().st_size >= size if prefix else path.stat().st_size == size,
            "R1 evidence size differs: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            require(bool(block), "Truncated R1 evidence")
            h.update(block); remaining -= len(block)
    require(h.hexdigest() == record["sha256"], "R1 evidence hash differs: " + str(path))
    return path


def effective_bindings(base, addendum_sha, code_sha):
    result = copy.deepcopy(base)
    result.update(code_sha256=code_sha, origin_code_sha256=base["code_sha256"],
                  engineering_repair_addendum_sha256=addendum_sha)
    return result


def validate_repair(protocol_path, authorization_path, config_path, output, repository,
                    addendum_path, acceptance_path):
    """No model imports, claims, directory creation or data reads before approval."""
    require(authorization_path is not None and Path(authorization_path).is_file(), "Original authorization absent")
    require(Path(addendum_path).is_file() and Path(acceptance_path).is_file(), "R1 addendum/acceptance absent")
    p, a, c = load(protocol_path), load(authorization_path), load(config_path)
    d, approval = load(addendum_path), load(acceptance_path)
    d_sha = file_sha(addendum_path); origin = d["origin"]
    require(d.get("schema") == SCHEMA and d.get("experiment") == EXPERIMENT, "Wrong R1 addendum")
    require(file_sha(protocol_path) == PROTOCOL_SHA == origin["protocol_sha256"], "Original G1 protocol changed")
    require(p["status"] == "PENDING_CHATGPT_FORMAL_TRAINING_GATE" and p["formal_training_authorized"] is False
            and p.get("pending_fields") == [], "Original public protocol changed")
    b0 = protocol_bindings(p, PROTOCOL_SHA)
    require(origin["bindings"] == b0 and origin["checkpoint"]["sha256"] == CHECKPOINT_SHA
            and origin["journal"]["sha256"] == JOURNAL_SHA and origin["journal"]["rows"] == 221,
            "R1 does not describe the sealed update105 fault")
    require(a.get("stage") == AUTH_STAGE and a.get("gate") == "CHATGPT_FORMAL_TRAINING_GATE_PASS"
            and a.get("authorized") is True and a.get("experiment") == EXPERIMENT
            and a.get("authorized_actions") == ACTIONS and a.get("bindings") == b0,
            "Original formal authorization differs")
    require(file_sha(authorization_path) == origin["authorization_file_sha256"]
            and digest(a) == origin["authorization_digest"] and a["execution_id"] == origin["execution_id"],
            "Original authorization identity differs")
    require(file_sha(config_path) == origin["runtime_config_sha256"] == a["runtime_config_sha256"],
            "Original runtime configuration changed")
    require(isinstance(c["val_authorization"], str)
            and file_sha(c["val_authorization"]) == origin["val_authorization_file_sha256"],
            "Original VAL authorization changed")
    output = Path(output)
    require(not output.is_symlink() and str(output.resolve()) == a["output"] == origin["output"], "R1 output relocated")
    require(str(control_anchor().resolve()) == a["control_anchor"] == origin["control_anchor"], "R1 anchor relocated")
    actual = source_catalog(repository)
    require(actual == d["code"]["files"] and digest(actual) == d["code"]["sha256"], "R1 production source changed")
    changes = {key: {"old_sha256": p["code"]["files"].get(key), "new_sha256": actual.get(key)}
               for key in set(actual) | set(p["code"]["files"])
               if p["code"]["files"].get(key) != actual.get(key)}
    require(set(changes) == CHANGED_FILES and changes == d["code"]["changes"], "Unapproved R1 code change")
    b1 = effective_bindings(b0, d_sha, d["code"]["sha256"])
    require(approval.get("schema") == "EXP17_J2_R1_ACCEPTANCE_V1" and approval.get("authorized") is True
            and approval.get("authorized_actions") == R1_ACTIONS
            and approval.get("historical_master_association_accepted") is True
            and approval.get("engineering_repair_addendum_sha256") == d_sha
            and approval.get("effective_bindings_json_digest") == digest(b1)
            and approval.get("execution_id") == a["execution_id"]
            and approval.get("output") == str(output.resolve())
            and approval.get("approval_basis") == "USER_APPROVED_R1_PLAN_IMPLEMENTATION_WITH_CONDITIONAL_POST_HOC_REUSE",
            "Exact R1 plan/evidence acceptance required")
    require(d["framework_sources_sha256"] == {"nn/modules/conv.py": CONV_SHA}, "R1 framework supplement differs")
    spec = importlib.util.find_spec("ultralytics")
    require(spec is not None and file_sha(Path(spec.origin).parent / "nn/modules/conv.py") == CONV_SHA,
            "Installed Conv implementation differs")
    verify_file(d["cpu_acceptance"])
    cpu = load(d["cpu_acceptance"]["path"])
    require(cpu["passed"] is True and cpu["production_catalog_sha256"] == d["code"]["sha256"],
            "R1 CPU acceptance does not cover these production bytes")
    for key in ("checkpoint", "latest_pointer", "claim", "seal"):
        verify_file(origin[key])
    verify_file(origin["journal"], prefix=True)
    anchor = verify_existing_claim(a, output)
    return p, a, c, d, approval, b0, b1, anchor


class R1Journal(RunJournal):
    """Exact pending-AUX capability; parent constructor remains unchanged."""
    def __init__(self, path, addendum, addendum_sha):
        self.path = Path(path)
        self.addendum_sha = addendum_sha
        origin = addendum["origin"]
        require(str(self.path.resolve()) == origin["journal"]["path"], "R1 journal relocated")
        verify_file(origin["journal"], prefix=True)
        self.rows, self.committed, self.pending, self.pending_aux = [], 0, None, None
        previous = "0" * 64
        for line in self.path.read_text(encoding="utf-8").splitlines(keepends=True):
            require(line.endswith("\n"), "Truncated R1 journal")
            row = json.loads(line); stored = row.pop("sha256")
            require(row["sequence"] == len(self.rows) + 1 and row["previous"] == previous
                    and digest(row) == stored, "R1 journal hash/order conflict")
            row["sha256"] = stored
            self._consume(row); self.rows.append(row); previous = stored
        require(len(self.rows) >= 221 and self.rows[218]["event"] == "UPDATE_COMMIT"
                and self.rows[218]["step"] == 105 and self.rows[218]["checkpoint_sha256"] == CHECKPOINT_SHA
                and self.rows[219]["event"] == "AUX_BEGIN" and self.rows[219]["epoch"] == 5
                and self.rows[220]["event"] == "HARD_GATE"
                and self.rows[220]["sha256"] == origin["journal"]["last_row_sha256"], "Wrong R1 fault boundary")
        require(self.pending is None, "AMBIGUOUS_INTERRUPTED_UPDATE: no replay")
        closed, stages, began = False, [], False
        for row in self.rows[221:]:
            event = row["event"]
            if event == "R1_BEGIN":
                require(not began and not closed and row["engineering_repair_addendum_sha256"] == addendum_sha,
                        "Duplicate/different R1 begin")
                began = True
            elif event == "R1_STAGE_COMMIT":
                require(began and not closed and len(stages) < len(STAGES)
                        and row["stage"] == STAGES[len(stages)]
                        and row["engineering_repair_addendum_sha256"] == addendum_sha, "R1 stage order differs")
                stages.append(row["stage"])
            elif event == "AUX_COMMIT" and not closed:
                require(began and stages == list(STAGES) and row["epoch"] == 5
                        and row["engineering_repair_addendum_sha256"] == addendum_sha, "R1 AUX commit without receipts")
                closed = True
            elif not closed:
                require(event == "R1_HARD_GATE", "Unexpected activity before R1 AUX commit")
        self.r1_closed = closed
        if not closed:
            require(self.committed == 105 and self.pending_aux == 5, "R1 only admits the original AUX5 boundary")
        else:
            require(self.pending_aux is None, "Later interrupted AUX has no R1 recovery capability")
        self.r1_stages = stages


def legacy_cache_resolver(addendum, b0, b1):
    allowed = {str(Path(r["directory"]).resolve()): r for r in addendum["legacy_caches"]}
    expected = {str(Path(addendum["origin"]["output"]) / "epoch_cache" / f"epoch_{e:03d}") for e in (3, 4)}
    require(set(allowed) == expected, "Unexpected legacy cache whitelist")
    for path, item in allowed.items():
        epoch = int(Path(path).name.split("_")[1])
        require(item["manifest_sha256"] == LEGACY_MANIFESTS[epoch], "R1 legacy cache differs from sealed evidence")
    def resolve(directory, manifest_sha):
        path = str(Path(directory).resolve())
        if path in allowed:
            require(manifest_sha == allowed[path]["manifest_sha256"], "Legacy cache manifest differs")
            return b0
        return b1
    return resolve


def publish(path, value):
    """Idempotent deterministic metadata; never overwrite a different artifact."""
    from tools.exp17.formal.checkpoint import atomic_json
    path = Path(path)
    require(not path.is_symlink(), "R1 artifact is a symlink")
    require(not list(path.parent.glob(path.name + ".tmp*")), "Partial R1 artifact must be reviewed")
    if path.exists():
        require(load(path) == value, "Existing R1 artifact differs: " + str(path))
    else:
        atomic_json(path, value)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def stage_commit(journal, stage, receipt, d_sha):
    existing = [r for r in journal.rows if r["event"] == "R1_STAGE_COMMIT" and r["stage"] == stage]
    if existing:
        require(len(existing) == 1 and existing[0]["receipt"] == receipt, "R1 stage receipt differs")
    else:
        count = sum(r["event"] == "R1_STAGE_COMMIT" for r in journal.rows)
        require(count < len(STAGES) and stage == STAGES[count], "R1 stage skipped")
        journal.append("R1_STAGE_COMMIT", stage=stage, receipt=receipt,
                       engineering_repair_addendum_sha256=d_sha)


def require_same_training_state(old, new):
    from tools.exp17.formal.checkpoint import state_hash
    allowed = {"bindings", "extra"}
    require(old.keys() == new.keys() and all(state_hash(old[k]) == state_hash(new[k])
            for k in old if k not in allowed), "R1 AUX changed committed training state")


def repair_marker(output, d, b1, acceptance_sha):
    return {"schema": "EXP17_J2_R1_ONCE_ONLY_REPAIR_CLAIM_V1", "execution_id": d["origin"]["execution_id"],
              "engineering_repair_addendum_sha256": b1["engineering_repair_addendum_sha256"],
              "acceptance_file_sha256": acceptance_sha, "origin_checkpoint_sha256": CHECKPOINT_SHA,
              "origin_journal_sha256": JOURNAL_SHA, "output": str(output.resolve()),
              "effective_bindings_json_digest": digest(b1), "budget_cap": 2100}


def begin_repair(output, journal, d, b1, acceptance_sha):
    marker_path = journal.path.parent / "R1_REPAIR_CLAIM.json"
    marker = repair_marker(output, d, b1, acceptance_sha)
    if marker_path.exists():
        require(not marker_path.is_symlink() and load(marker_path) == marker,
                "R1 repair already claimed under different identity")
    else:
        require(len(journal.rows) == 221, "R1 claim missing after recovery activity")
        exclusive_json(marker_path, marker)
    if not any(r["event"] == "R1_BEGIN" for r in journal.rows):
        journal.append("R1_BEGIN", engineering_repair_addendum_sha256=b1["engineering_repair_addendum_sha256"],
                       repair_claim_sha256=file_sha(marker_path), origin_journal_sha256=JOURNAL_SHA)


def validate_ema_completion(receipt, adapter, output, d_sha, canonical):
    from tools.exp17.formal.repair_evidence import validate_evaluation_pair
    expected = {str((output / "VAL_epoch005_ema.json").resolve())}
    ledger = Path(adapter.config["val_ledger_root"])
    val_output = Path(adapter.config["val_output"])
    for mode in ("standard", "fixed"):
        expected.update((str((ledger / f"epoch005_ema_{mode}.json").resolve()),
                         str((val_output / f"epoch005_ema_{mode}" / "evaluation.json").resolve())))
    require(receipt.get("schema") == "EXP17_J2_R1_EMA_ROLE_COMPLETION_V1"
            and receipt.get("epoch") == 5 and receipt.get("role") == "ema"
            and receipt.get("engineering_repair_addendum_sha256") == d_sha
            and receipt.get("native_state_sha256") == canonical["ema"]
            and receipt.get("origin_checkpoint_sha256") == CHECKPOINT_SHA
            and receipt.get("isolation_passed") is True and receipt.get("native_passes_new") == 2,
            "EMA completion identity differs")
    require(len(receipt.get("artifacts", [])) == 5
            and {item["path"] for item in receipt["artifacts"]} == expected,
            "EMA completion must bind the exact five artifacts")
    for item in receipt["artifacts"]:
        verify_file(item)
    ema = load(output / "VAL_epoch005_ema.json")
    for mode in ("standard", "fixed"):
        require(load(ledger / f"epoch005_ema_{mode}.json") == {
            "epoch":5, "role":"ema", "mode":mode, "protocol_sha256":PROTOCOL_SHA,
            "state":"ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY"}, "EMA pass claim differs")
        require(ema[mode] == load(val_output / f"epoch005_ema_{mode}" / "evaluation.json"),
                "EMA role differs from its raw pass result")
    fitness = validate_evaluation_pair(ema["standard"], ema["fixed"])
    require(ema.get("role") == "ema" and ema.get("fitness") == fitness, "EMA role/fitness differs")
    return ema


def verify_repair_receipts(journal, d, b1, output, acceptance_sha, config):
    """Verify immutable migration evidence before any subsequent runtime import."""
    stages = [r for r in journal.rows if r["event"] == "R1_STAGE_COMMIT"]
    beginnings = [r for r in journal.rows if r["event"] == "R1_BEGIN"]
    if not beginnings:
        require(not stages and len(journal.rows) == 221, "R1 evidence exists without its begin")
        return
    marker_path = journal.path.parent / "R1_REPAIR_CLAIM.json"
    require(marker_path.is_file() and not marker_path.is_symlink()
            and load(marker_path) == repair_marker(output, d, b1, acceptance_sha)
            and file_sha(marker_path) == beginnings[0]["repair_claim_sha256"], "R1 repair claim changed")
    names = dict(zip(STAGES, ("historical_master_association.json", "ema_role_completion.json",
                              "best_epoch005.json", "checkpoint_epoch005.json")))
    for row in stages:
        expected = output / "r1_provenance" / names[row["stage"]]
        require(row["receipt"]["path"] == str(expected.resolve()), "R1 stage receipt relocated")
        verify_file(row["receipt"])
    if any(r["stage"] == "EMA_ROLE" for r in stages):
        from types import SimpleNamespace
        receipt = load(output / "r1_provenance" / names["EMA_ROLE"])
        validate_ema_completion(receipt, SimpleNamespace(config=config), output,
                                b1["engineering_repair_addendum_sha256"], d["origin"]["canonical_native_sha256"])
    if journal.r1_closed:
        close = next(r for r in journal.rows[221:] if r["event"] == "AUX_COMMIT")
        ready = load(output / "r1_provenance" / names["CHECKPOINT_READY"])
        expected_active = {"stage":"EXP17_J2_R1_ACTIVE_RECOVERY_V1",
            "directory":str((output / "recovery_r1_v1").resolve()), "bindings":b1,
            "engineering_repair_addendum_sha256":b1["engineering_repair_addendum_sha256"],
            "checkpoint_sha256_at_activation":ready["last"]["sha256"], "aux_commit_sequence":close["sequence"]}
        require(load(output / "R1_ACTIVE_RECOVERY.json") == expected_active
                and close["checkpoint_sha256"] == ready["last"]["sha256"], "R1 activation/journal differs")
        periodic = ready["periodic"]
        directory = output / "recovery_r1_v1"
        require(periodic["file"] == "epoch005_full_recovery.pt"
                and load(directory / (periodic["file"] + ".json")) == periodic, "R1 periodic receipt changed")
        verify_file({"path":str(directory / periodic["file"]), "bytes":periodic["bytes"], "sha256":periodic["sha256"]})


def resume_epoch5(adapter, protocol, d, b0, b1, output, journal, old_store, store, acceptance_sha):
    from tools.exp17.formal.checkpoint import capture_training_state, restore_training_state
    from tools.exp17.formal.exports import save_best_bundle
    from tools.exp17.formal.sampler import SealedEpochCache
    from tools.exp17.formal.validation import best_update
    from tools.exp17.formal.repair_evidence import verify_historical_master
    from tools.exp14.formal_e0_e1 import hash_state
    d_sha = b1["engineering_repair_addendum_sha256"]
    provenance = output / "r1_provenance"
    provenance.mkdir(exist_ok=True)
    require(not provenance.is_symlink(), "R1 provenance path changed")
    begin_repair(output, journal, d, b1, acceptance_sha)
    from tools.exp17.formal.executor import _rows
    _rows(output / "steps.jsonl", 105)
    old, _ = old_store.load_latest(expected_sha256=CHECKPOINT_SHA)
    require(old["position"] == {"epoch": 4, "batch_position": 21, "global_update": 105}
            and old["extra"] == {"best": {}, "epoch_events_complete": False}, "Unexpected original recovery position")
    position, sampler, _ = restore_training_state(adapter.trainer, old, bindings=b0)
    adapter.trainer.epoch = position["epoch"]
    cache = SealedEpochCache.open(sampler["cache_directory"], bindings=b0,
                                expected_manifest_sha256=sampler["cache_manifest_sha256"])
    cursor = cache.cursor(); cursor.load_state_dict(sampler); adapter.set_sampler(cursor)
    canonical = {role: hash_state({k:v for k,v in old[field].items() if not k.startswith("region_projection.")})
                 for role,field in (("master","state"),("ema","ema"))}
    association = verify_historical_master(d, protocol, adapter.config, canonical)
    association_receipt = publish(provenance / "historical_master_association.json", association)
    stage_commit(journal, "MASTER_ASSOCIATION", association_receipt, d_sha)
    master = copy.deepcopy(association["master_evaluation"])
    master["recovery_provenance"] = {"kind": "POST_HOC_FROM_SEALED_ARTIFACTS", **association_receipt}
    publish(output / "VAL_epoch005_master.json", master)
    ema_receipt_path = provenance / "ema_role_completion.json"
    ema_output_path = output / "VAL_epoch005_ema.json"
    if ema_receipt_path.exists():
        ema_receipt = load(ema_receipt_path)
        ema = validate_ema_completion(ema_receipt, adapter, output, d_sha, canonical)
    else:
        ledger = Path(adapter.config["val_ledger_root"])
        require(not any((ledger / f"epoch005_ema_{mode}.json").exists() for mode in ("standard", "fixed"))
                and not ema_output_path.exists(), "Claimed/partial EMA has no accepted completion receipt; no retry")
        ema = adapter.evaluate(5, "ema")
        ema_role = publish(ema_output_path, ema)
        artifacts = [ema_role]
        for mode in ("standard", "fixed"):
            for path in (ledger / f"epoch005_ema_{mode}.json",
                         Path(adapter.config["val_output"]) / f"epoch005_ema_{mode}" / "evaluation.json"):
                artifacts.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha(path)})
        ema_receipt = {"schema": "EXP17_J2_R1_EMA_ROLE_COMPLETION_V1", "role": "ema", "epoch": 5,
                       "engineering_repair_addendum_sha256": d_sha, "native_state_sha256": canonical["ema"],
                       "origin_checkpoint_sha256": CHECKPOINT_SHA, "isolation_passed": True,
                       "native_passes_new": 2, "artifacts": artifacts}
        publish(ema_receipt_path, ema_receipt)
        validate_ema_completion(ema_receipt, adapter, output, d_sha, canonical)
    stage_commit(journal, "EMA_ROLE", {"path": str(ema_receipt_path), "bytes": ema_receipt_path.stat().st_size,
                                        "sha256": file_sha(ema_receipt_path)}, d_sha)
    # A derived payload records new serialization/execution code, not a claim
    # that C1 produced the first105 physical optimizer updates.
    base = capture_training_state(adapter.trainer, bindings=b1, position=position, sampler_state=cursor.state_dict(),
                                  extra={"best": {}, "epoch_events_complete": False})
    require_same_training_state(old, base)
    best = {}
    for role, evaluation in (("master", master), ("ema", ema)):
        record, selected = best_update(None, epoch=5, role=role, master_sha256=canonical["master"],
                                       ema_sha256=canonical["ema"], evaluation=evaluation)
        require(selected, "Epoch5 must be the first accepted Best")
        pointer = output / "best" / f"best_{role}.json"
        history = output / "best" / f"best_{role}_epoch005.selection.json"
        if pointer.exists() or history.exists():
            require(pointer.exists() and history.exists() and load(pointer) == load(history),
                    "Partial Best transaction; preserve and HOLD")
            receipt = load(pointer)
            verify_file({"path": str(pointer.parent / receipt["file"]), "bytes": receipt["bytes"], "sha256": receipt["sha256"]})
            import torch
            bundle = torch.load(pointer.parent / receipt["file"], map_location="cpu", weights_only=True)
            from tools.exp17.formal.checkpoint import state_hash
            require(bundle["bindings"] == b1 and bundle["selection"] == record
                    and state_hash(bundle["state"]) == old["master_state_sha256"]
                    and state_hash(bundle["ema"]) == old["ema_state_sha256"]
                    and bundle["master_native_state_sha256"] == canonical["master"]
                    and bundle["ema_native_state_sha256"] == canonical["ema"]
                    and receipt["bindings_sha256"] == state_hash(b1)
                    and receipt["role"] == "best_" + role and receipt["epoch"] == 5
                    and receipt["schema"] == "EXP17_G1_BEST_FP32_TENSOR_BUNDLE_V1", "Existing Best differs")
        else:
            require(not list((output / "best").glob(f"best_{role}_slot*.pt")), "Orphan Best blob; no overwrite")
            receipt = save_best_bundle(output / "best", base, record)
        record["checkpoint"] = receipt; best[role] = record
    best_receipt = publish(provenance / "best_epoch005.json", best)
    stage_commit(journal, "BEST_READY", best_receipt, d_sha)
    extra = {"best": best, "epoch_events_complete": True,
             "engineering_repair": {"engineering_repair_addendum_sha256": d_sha,
                 "origin_checkpoint_sha256": CHECKPOINT_SHA, "origin_journal_sha256": JOURNAL_SHA,
                 "original_updates": [1, 105], "historical_master_association": association_receipt}}
    payload = capture_training_state(adapter.trainer, bindings=b1, position=position,
                                     sampler_state=cursor.state_dict(), extra=extra)
    require_same_training_state(old, payload)
    if (store.directory / "latest.json").exists():
        restored, receipt = store.load_latest()
        from tools.exp17.formal.checkpoint import state_hash
        require(state_hash(restored) == state_hash(payload), "R1 Last differs before AUX commit")
    else:
        require(not list(store.directory.glob("*.pt*")), "Orphan R1 checkpoint; no overwrite")
        receipt = store.save_latest(payload)
    name = "epoch005_full_recovery.pt"
    if (store.directory / name).exists() or (store.directory / (name + ".json")).exists():
        periodic, periodic_receipt = store.load_artifact(name)
        from tools.exp17.formal.checkpoint import state_hash
        require(state_hash(periodic) == state_hash(payload), "Existing epoch5 periodic differs")
    else:
        periodic_receipt = store.save_artifact(name, payload, role="periodic")
    checkpoint_receipt = publish(provenance / "checkpoint_epoch005.json", {"last": receipt, "periodic": periodic_receipt})
    stage_commit(journal, "CHECKPOINT_READY", checkpoint_receipt, d_sha)
    active = {"stage": "EXP17_J2_R1_ACTIVE_RECOVERY_V1", "directory": str(store.directory.resolve()),
              "bindings": b1, "engineering_repair_addendum_sha256": d_sha,
              "checkpoint_sha256_at_activation": receipt["sha256"], "aux_commit_sequence": len(journal.rows)+1}
    publish(output / "R1_ACTIVE_RECOVERY.json", active)
    journal.append("AUX_COMMIT", epoch=5, checkpoint_sha256=receipt["sha256"],
                   engineering_repair_addendum_sha256=d_sha, recovery_directory=str(store.directory.resolve()),
                   checkpoint_receipt=checkpoint_receipt)
    require(journal.pending_aux is None and journal.committed == 105, "R1 did not close exact AUX5")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "authorization", "runtime-config", "output", "engineering-addendum", "repair-acceptance"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="read-only validation, no claims/model imports")
    parser.add_argument("--resume-authorization", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    p, a, c, d, approval, b0, b1, anchor = validate_repair(args.protocol, args.authorization, args.runtime_config,
        args.output, repo, args.engineering_addendum, args.repair_acceptance)
    d_sha = file_sha(args.engineering_addendum)
    journal = R1Journal(anchor / "updates.jsonl", d, d_sha)
    verify_repair_receipts(journal, d, b1, args.output, file_sha(args.repair_acceptance), c)
    cache_resolver = legacy_cache_resolver(d, b0, b1)
    if not journal.r1_closed:
        from tools.exp17.formal.executor import _rows
        from tools.exp17.formal.repair_evidence import verify_historical_master
        _rows(args.output / "steps.jsonl", 105)
        verify_historical_master(d, p, c, d["origin"]["canonical_native_sha256"])
        for item in d["legacy_caches"]:
            manifest = Path(item["directory"]) / "manifest.json"
            require(not manifest.is_symlink() and file_sha(manifest) == item["manifest_sha256"]
                    and load(manifest)["bindings"] == b0, "R1 legacy cache changed before EMA")
    require(not any(r["event"] == "TRAIN_COMPLETE" for r in journal.rows), "Completed experiment cannot restart")
    if args.preflight_only:
        print(json.dumps({"stage": "EXP17_J2_R1_READONLY_PREFLIGHT", "passed": True,
                          "committed": journal.committed, "pending_aux": journal.pending_aux,
                          "effective_code_sha256": b1["code_sha256"]}), flush=True)
        return
    with RunLease(anchor):
        journal = R1Journal(anchor / "updates.jsonl", d, d_sha)
        verify_repair_receipts(journal, d, b1, args.output, file_sha(args.repair_acceptance), c)
        from tools.exp17.formal.checkpoint import AtomicCheckpointStore
        from tools.exp17.formal.monitor import ensure_space
        ensure_space(args.output, required_bytes=p["resources"]["before_epoch_min_free_bytes"] + d["retained_origin_recovery_bytes"])
        old_store = AtomicCheckpointStore(args.output / "recovery", b0)
        store = AtomicCheckpointStore(args.output / "recovery_r1_v1", b1)
        recovered = None
        if journal.r1_closed:
            require(args.resume_authorization is not None, "Exact committed-state R1 resume authorization required")
            recovered, receipt = store.load_latest()
            last = next(r for r in reversed(journal.rows) if r["event"] in ("UPDATE_COMMIT", "AUX_COMMIT"))
            gate = load(args.resume_authorization)
            require(receipt["sha256"] == last["checkpoint_sha256"] and gate == {
                "schema": "EXP17_J2_R1_COMMITTED_RESUME_V1", "authorization_digest": digest(a),
                "engineering_repair_addendum_sha256": d_sha, "checkpoint_sha256": receipt["sha256"],
                "journal_sha256": file_sha(journal.path), "committed_update": journal.committed,
                "output": str(args.output.resolve())}, "R1 committed resume identity differs")
        else:
            from tools.exp17.formal.executor import _rows
            _rows(args.output / "steps.jsonl", 105)
            allowed_claims = {f"epoch005_{role}_{mode}.json" for role in ("master", "ema")
                              for mode in ("standard", "fixed")}
            require({x.name for x in Path(c["val_ledger_root"]).iterdir()} <= allowed_claims,
                    "Unexpected VAL attempt before R1 AUX recovery")
            begin_repair(args.output, journal, d, b1, file_sha(args.repair_acceptance))
        from tools.exp17.formal.native_adapter import NativeAdapter
        adapter = None
        try:
            adapter = NativeAdapter(c, p, args.output, b1)
            if recovered is None:
                recovered = resume_epoch5(adapter, p, d, b0, b1, args.output, journal, old_store, store,
                                          file_sha(args.repair_acceptance))
            from tools.exp17.formal.executor import run_session
            result = run_session(adapter, p, b1, args.output, journal, store, recovered=recovered,
                                 legacy_cache_bindings=cache_resolver)
            exclusive_json(args.output / "completion.json", result)
        except BaseException as error:
            evidence = {"status": "EXP17_J2_R1_HARD_GATE", "error_type": type(error).__name__,
                        "message": str(error), "traceback": traceback.format_exc(), "updates": journal.committed,
                        "engineering_repair_addendum_sha256": d_sha}
            try:
                journal.append("R1_HARD_GATE", error_type=type(error).__name__, message=str(error),
                               engineering_repair_addendum_sha256=d_sha)
                exclusive_json(args.output / f"r1_hard_gate_{len(journal.rows):05d}.json", evidence)
            except BaseException:
                print(json.dumps(evidence), flush=True)
            raise
        finally:
            if adapter is not None:
                adapter.close()


if __name__ == "__main__":
    main()
