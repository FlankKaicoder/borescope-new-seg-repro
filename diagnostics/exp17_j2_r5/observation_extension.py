"""Versioned, identity-bound continuation of R5's committed Epoch5 observation.

The user approved observing the same run through Epoch10, then applying the
original scientific decision rule. Shared training tools and original records
remain frozen. This entry owns only its addendum, migration and new namespaces.
"""
import argparse
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from tools.exp17.formal.authorization import (RunJournal, RunLease, digest,
    exclusive_json, file_sha, load, require, source_catalog)
from tools.exp17.formal import repaired_seed42 as r5

SCHEMA = "EXP17_J2_R5_E10_OBSERVATION_V1"
ROOT = Path("/root/exp17_r5_e10_observation_v1")
RUN, ANCHOR = r5.ROOT, r5.ANCHOR
GATE = Path("/root/exp17_r5_gate_20261124_v5")
ORIGIN_CODE = "928f2726b9ca07bf44c1d1eee14beda25036c4e8957b5afd96f14a3fc0c17e07"
ORIGIN_CHECKPOINT = "9193e57e2a31b7b7dafdfece107ddbce8f6f10a97a70390865cfcefcda5784ad"
ORIGIN_JOURNAL = "43c0acde2747c0d829b44786f03792a2e0455b5b8040d3809a0dfd6709076fb6"
MARKER_NAME = "R5_E10_OBSERVATION_EXTENSION_CLAIM.json"
ACTIVE_NAME = "R5_E10_OBSERVATION_ACTIVE_STORE.json"
SCOPE = {"origin_update": 105, "observation_epoch": 10, "observation_update": 210,
         "first_new_update": 106, "conditional_first_update": 211,
         "maximum_effective_updates": 2100, "maximum_native_VAL_passes": 80,
         "accept_original_epoch5_HOLD_for_observation_only": True,
         "E10_PASS": "RELOAD_SAME210_AND_CONTINUE_TO2100", "E10_HOLD": "STOP_AT210",
         "original_scientific_parameters_unchanged": True}


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Missing or symlink extension evidence")
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def verify_file(item, *, prefix=False):
    path = Path(item["path"])
    require(path.is_file() and not path.is_symlink(), "Missing extension evidence: " + str(path))
    size = item["bytes"]
    require(path.stat().st_size >= size if prefix else path.stat().st_size == size,
            "Extension evidence size differs: " + str(path))
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            require(bool(block), "Truncated extension evidence")
            hasher.update(block)
            remaining -= len(block)
    require(hasher.hexdigest() == item["sha256"], "Extension evidence SHA differs: " + str(path))
    return path


def extended_catalog(repository):
    repository = Path(repository).resolve()
    files = source_catalog(repository)
    own = Path(__file__).resolve()
    require(own.is_relative_to(repository), "Extension source is outside its repository")
    files[own.relative_to(repository).as_posix()] = file_sha(own)
    return files


def validate_extended_catalog(repository, code):
    actual = extended_catalog(repository)
    require(actual == code["files"] and digest(actual) == code["sha256"], "Observation extension source changed")


def effective_bindings(base, addendum_sha, code_sha):
    result = copy.deepcopy(base)
    result.update(code_sha256=code_sha, origin_code_sha256=base["code_sha256"],
                  engineering_observation_addendum_sha256=addendum_sha)
    return result


def migrate_payload(payload, base, effective, addendum_sha):
    from tools.exp17.formal.checkpoint import state_hash, validate_payload
    validate_payload(payload, base)
    require(effective == effective_bindings(base, addendum_sha, effective["code_sha256"]),
            "Observation migration changed science/data bindings")
    require(payload["position"] == {"epoch": 4, "batch_position": 21, "global_update": 105}
            and payload["extra"].get("epoch_events_complete") is True, "Migration is only for complete105")
    require("engineering_observation" not in payload["extra"], "Payload was already migrated")
    derived = copy.deepcopy(payload)
    derived["bindings"] = copy.deepcopy(effective)
    derived["extra"]["engineering_observation"] = {"addendum_sha256": addendum_sha,
        "origin_bindings_sha256": digest(base), "origin_update": 105}
    for key in payload:
        if key not in ("bindings", "extra"):
            require(state_hash(payload[key]) == state_hash(derived[key]), "Migration changed training field: " + key)
    require(all(state_hash(value) == state_hash(derived["extra"][key]) for key, value in payload["extra"].items()),
            "Migration changed original auxiliary metadata")
    validate_payload(derived, effective)
    return derived


def verify_existing_claim(authorization):
    """Exact original R5 claim format, with the original R5 anchor and seal."""
    expected = {"experiment": r5.EXPERIMENT, "authorization_sha256": digest(authorization),
                "execution_id": authorization["execution_id"], "output": str(RUN), "cap": 2100}
    require(load(ANCHOR / "CLAIM.json") == load(RUN / "EXECUTION_SEAL.json") == expected,
            "Original R5 permanent claim/seal differ")
    require((ANCHOR / "updates.jsonl").is_file(), "Original permanent journal missing")
    return ANCHOR


def validate_observation_boundary(payload, journal, update, epoch, bindings):
    from tools.exp17.formal.checkpoint import validate_payload
    validate_payload(payload, bindings)
    require(update == epoch * 21 and journal.committed == update
            and journal.pending is None and journal.pending_aux is None, "Observation boundary has incomplete work")
    require(payload["position"] == {"epoch": epoch - 1, "batch_position": 21, "global_update": update}
            and payload["extra"].get("epoch_events_complete") is True, "Observation checkpoint is not AUX-complete")
    stateful = [r for r in journal.rows if r["event"] in ("UPDATE_COMMIT", "AUX_COMMIT")]
    require(stateful and stateful[-1]["event"] == "AUX_COMMIT" and stateful[-1]["epoch"] == epoch,
            "Observation journal is not at committed AUX boundary")


def legacy_cache_bindings(directory, manifest_sha, addendum, base, effective):
    directory = Path(directory).resolve()
    cache_root = Path(addendum["output"]).resolve() / "epoch_cache"
    require(directory.parent == cache_root, "Cache escaped original output")
    allowed = addendum["legacy_caches"]
    matches = [r for r in allowed if Path(r["path"]).resolve() == directory]
    if matches:
        require(len(matches) == 1, "Duplicate old-cache identity")
        item = matches[0]
        require(item["manifest_sha256"] == manifest_sha and item["bindings"] == base
                and item["epoch"] in (3, 4) and directory.name == f"epoch_{item['epoch']:03d}",
                "Old cache is outside exact observation allowlist")
        return base
    require(directory.parent == cache_root and directory.name.startswith("epoch_"), "Cache escaped original output")
    try:
        epoch = int(directory.name.removeprefix("epoch_"))
    except ValueError:
        raise RuntimeError("Invalid future epoch-cache name") from None
    require(directory.name == f"epoch_{epoch:03d}" and epoch >= 5, "Unbound legacy epoch cache")
    return effective


@contextmanager
def best_namespace(output, bindings, best_directory=None):
    from tools.exp17.formal import exports
    output = Path(output).resolve()
    destination = Path(best_directory).resolve() if best_directory is not None else output / "best_e10_observation_v1"
    require(destination.parent == output and destination.name == "best_e10_observation_v1", "Unexpected Best namespace")
    original = exports.save_best_bundle
    def save(requested, payload, selection_record, **kwargs):
        require(Path(requested).resolve() == output / "best" and payload["bindings"] == bindings,
                "Unexpected Best write or binding in observation extension")
        return original(destination, payload, selection_record, **kwargs)
    with patch.object(exports, "save_best_bundle", side_effect=save):
        yield


def check_e10_evidence(output, anchor, payload, journal, store, bindings):
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp17.formal.checkpoint import state_hash
    from tools.exp17.formal.r5_gate import decide
    output, anchor = Path(output), Path(anchor)
    validate_observation_boundary(payload, journal, 210, 10, bindings)
    periodic, _ = store.load_artifact("epoch010_full_recovery.pt")
    require(state_hash(periodic) == state_hash(payload), "Epoch10 periodic and Last differ")
    wanted = {f"epoch{epoch:03d}_{role}_{mode}.json" for epoch in (5, 10)
              for role in ("master", "ema") for mode in ("standard", "fixed")}
    require({p.name for p in (anchor / "val_passes").glob("*.json")} == wanted, "Epoch10 must have exactly8 unique passes")
    artifacts = {}
    for epoch in (5, 10):
        for role, field in (("master", "state"), ("ema", "ema")):
            role_path = output / f"VAL_epoch{epoch:03d}_{role}.json"
            role_result = load(role_path)
            artifacts[str(role_path)] = file_sha(role_path)
            expected_state = hash_state({k: v for k, v in payload[field].items() if not k.startswith("region_projection.")}) if epoch == 10 else None
            pair_states = []
            for mode, confidence in (("standard", .001), ("fixed", .25)):
                name = f"epoch{epoch:03d}_{role}_{mode}"
                directory = output / "val" / name
                receipt_path, raw_path = directory / "pass_receipt.json", directory / "evaluation.json"
                receipt = load(receipt_path)
                claim_path = anchor / "val_passes" / (name + ".json")
                claim = load(claim_path)
                require((claim["epoch"], claim["role"], claim["mode"]) == (epoch, role, mode)
                        and claim["protocol_sha256"] == bindings["protocol_sha256"]
                        and claim["state"] == "ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY", "Observation VAL claim differs")
                require((receipt["epoch"], receipt["role"], receipt["mode"]) == (epoch, role, mode)
                        and receipt["confidence"] == confidence
                        and receipt["protocol_sha256"] == bindings["protocol_sha256"]
                        and receipt["evaluation_sha256"] == file_sha(raw_path)
                        and receipt["result_digest"] == digest(role_result[mode]), "Observation VAL evidence differs")
                if expected_state is not None:
                    require(receipt["native_state_sha256"] == expected_state, "Epoch10 VAL used another model")
                pair_states.append(receipt["native_state_sha256"])
                for path in (receipt_path, raw_path, claim_path):
                    artifacts[str(path)] = file_sha(path)
            require(pair_states[0] == pair_states[1], "VAL roles used inconsistent states across confidence modes")
    diagnostic_path = output / "diagnostics_epoch010.json"
    result = decide(load(diagnostic_path), load(output / "VAL_epoch010_master.json"))
    artifacts[str(diagnostic_path)] = file_sha(diagnostic_path)
    result["artifacts"] = artifacts
    return result


def epoch_guard(epoch, journal, store, *, root, addendum, addendum_sha, bindings):
    if epoch < 10:
        return None
    root = Path(root)
    path = root / "epoch010_gate.json"
    if not path.exists():
        require(epoch == 10 and journal.committed == 210, "Missing Epoch10 observation decision")
        payload, receipt = store.load_latest()
        result = check_e10_evidence(Path(addendum["output"]), ANCHOR, payload, journal, store, bindings)
        result.update(checkpoint=receipt, bindings=bindings, addendum_sha256=addendum_sha,
                      journal_last_sha256=journal.rows[-1]["sha256"], journal_prefix_sha256=file_sha(journal.path), updates=210)
        exclusive_json(path, result)
        return {"status": "R5_EPOCH10_" + result["status"], "updates": 210}
    result = load(path)
    require(result["bindings"] == bindings and result["addendum_sha256"] == addendum_sha, "Epoch10 decision binding differs")
    require(result["status"] == "PASS", "Epoch10 scientific HOLD forbids update211")
    require(any(r["sha256"] == result["journal_last_sha256"] and r["event"] == "AUX_COMMIT" and r["epoch"] == 10
                for r in journal.rows), "Epoch10 decision is outside journal lineage")
    for artifact, expected in result["artifacts"].items():
        require(file_sha(artifact) == expected, "Accepted Epoch10 evidence changed")
    if journal.committed == 210:
        _, receipt = store.load_latest()
        require(receipt == result["checkpoint"], "Epoch10 continuation checkpoint differs")
    return None


def validate_active_store(marker, journal, addendum_sha, bindings, store_directory):
    require(marker["schema"] == SCHEMA + "_ACTIVE_STORE" and marker["addendum_sha256"] == addendum_sha
            and marker["bindings_sha256"] == digest(bindings)
            and marker["store_directory"] == str(Path(store_directory).resolve()), "Active observation store differs")
    extension_root = Path(marker["extension_root"])
    require(extension_root.is_absolute() and str(extension_root.resolve()) == str(extension_root)
            and marker["completion_path"] == str(extension_root / "completion.json")
            and marker["epoch10_gate_path"] == str(extension_root / "epoch010_gate.json"), "Active observation result paths differ")
    rows = [r for r in journal.rows if r["event"] == "SOURCE_MIGRATION_COMMIT"]
    require(len(rows) == 1, "Missing or duplicate observation source migration")
    row = rows[0]
    require(row["sha256"] == marker["migration_event_sha256"]
            and row["addendum_sha256"] == addendum_sha and row["bindings_sha256"] == digest(bindings)
            and row["store_directory"] == marker["store_directory"]
            and row["best_directory"] == marker["best_directory"]
            and all(row[key] == marker[key] for key in ("extension_root", "completion_path", "epoch10_gate_path"))
            and row["checkpoint_sha256"] == marker["migration_checkpoint_sha256"], "Active store has no matching journal commit")


def prepare(repository, root):
    from tools.exp17.formal.r5_gate import RULE
    protocol, config, authorization, base = r5.validate(repository, GATE)
    verify_existing_claim(authorization)
    journal = RunJournal(ANCHOR / "updates.jsonl")
    require(journal.committed == 105 and len(journal.rows) == 221
            and file_sha(journal.path) == ORIGIN_JOURNAL and base["code_sha256"] == ORIGIN_CODE,
            "Observation preparation requires the original committed105 snapshot")
    early = load(RUN / "epoch005_gate.json")
    pointer = load(RUN / "recovery/latest.json")
    require(early["status"] == "HOLD" and early["updates"] == 105 and early["mismatched_layers"] == []
            and early["checkpoint"] == pointer and pointer["sha256"] == ORIGIN_CHECKPOINT,
            "Wrong Epoch5 observation source")
    root = Path(root).resolve()
    require(root != RUN.resolve() and not root.is_relative_to(RUN.resolve()), "Extension metadata intersects formal output")
    root.mkdir(parents=True, exist_ok=True)
    code = {"files": extended_catalog(repository)}
    code["sha256"] = digest(code["files"])
    cpu_path = root / "cpu_tests.json"
    cpu = load(cpu_path)
    require(cpu["status"] == "PASS" and cpu["code_sha256"] == code["sha256"], "Targeted CPU tests do not bind C6")
    immutable_paths = [GATE / name for name in ("protocol.json", "runtime_config.json", "formal_authorization.json", "val_authorization.json")]
    immutable_paths += [ANCHOR / "CLAIM.json", RUN / "EXECUTION_SEAL.json", RUN / "epoch005_gate.json", RUN / "completion.json"]
    immutable_paths += list((RUN / "recovery").glob("*")) + list((RUN / "best").glob("*"))
    immutable_paths += [RUN / "diagnostics_epoch005.json", RUN / "VAL_epoch005_master.json", RUN / "VAL_epoch005_ema.json"]
    immutable_paths += list((ANCHOR / "val_passes").glob("*.json"))
    immutable_paths += list((RUN / "val").glob("*/evaluation.json")) + list((RUN / "val").glob("*/pass_receipt.json"))
    require(len(list((ANCHOR / "val_passes").glob("*.json"))) == 4, "Original pass budget differs")
    legacy = []
    for epoch in (3, 4):
        directory = RUN / "epoch_cache" / f"epoch_{epoch:03d}"
        manifest = load(directory / "manifest.json")
        require(manifest["epoch"] == epoch and manifest["bindings"] == base, "Original cache bindings differ")
        legacy.append({"path": str(directory.resolve()), "manifest_sha256": file_sha(directory / "manifest.json"),
                       "epoch": epoch, "bindings": base})
    addendum = {"schema": SCHEMA, "experiment": r5.EXPERIMENT, "extension_root": str(root), "output": str(RUN),
        "control_anchor": str(ANCHOR), "store_directory": str(RUN / "recovery_e10_observation_v1"),
        "best_directory": str(RUN / "best_e10_observation_v1"), "scope": SCOPE, "E10_rule": RULE,
        "completion_path": str(root / "completion.json"), "epoch10_gate_path": str(root / "epoch010_gate.json"),
        "origin_bindings": base, "origin_execution_id": authorization["execution_id"],
        "origin_authorization_digest": digest(authorization), "origin_checkpoint": record(RUN / "recovery" / pointer["file"]),
        "origin_latest_pointer": pointer, "origin_journal": {**record(journal.path), "rows": 221,
                                                               "last_row_sha256": journal.rows[-1]["sha256"]},
        "immutable_original_files": [record(p) for p in sorted(set(immutable_paths))],
        "legacy_caches": legacy, "code": code, "cpu_acceptance": record(cpu_path)}
    addendum_path = root / "engineering_observation_addendum.json"
    exclusive_json(addendum_path, addendum)
    effective = effective_bindings(base, file_sha(addendum_path), code["sha256"])
    exclusive_json(root / "approval.json", {"schema": SCHEMA + "_APPROVAL", "authorized": True,
        "basis": "USER_EXPLICITLY_APPROVED_VERSIONED_E5_TO_E10_OBSERVATION_AND_CONDITIONAL_CONTINUATION",
        "addendum_sha256": file_sha(addendum_path), "effective_bindings_digest": digest(effective),
        "execution_id": authorization["execution_id"], "output": str(RUN), "scope": SCOPE})
    print(json.dumps({"status": "OBSERVATION_METADATA_READY", "code_sha256": code["sha256"], "formal_mutations": 0}), flush=True)


def validate(repository, root):
    protocol, config, authorization, base = r5.validate(repository, GATE)
    verify_existing_claim(authorization)
    root = Path(root).resolve()
    addendum_path = root / "engineering_observation_addendum.json"
    addendum, approval = load(addendum_path), load(root / "approval.json")
    addendum_sha = file_sha(addendum_path)
    from tools.exp17.formal.r5_gate import RULE
    require(addendum["schema"] == SCHEMA and addendum["experiment"] == r5.EXPERIMENT
            and addendum["scope"] == SCOPE and addendum["E10_rule"] == RULE,
            "Observation scope or scientific decision rule changed")
    require(addendum["extension_root"] == str(root) and addendum["output"] == str(RUN)
            and addendum["control_anchor"] == str(ANCHOR)
            and addendum["store_directory"] == str(RUN / "recovery_e10_observation_v1")
            and addendum["best_directory"] == str(RUN / "best_e10_observation_v1")
            and addendum["completion_path"] == str(root / "completion.json")
            and addendum["epoch10_gate_path"] == str(root / "epoch010_gate.json"), "Observation paths changed")
    require(addendum["origin_bindings"] == base and base["code_sha256"] == ORIGIN_CODE
            and addendum["origin_execution_id"] == authorization["execution_id"]
            and addendum["origin_authorization_digest"] == digest(authorization), "Observation origin changed")
    require(addendum["origin_checkpoint"]["sha256"] == ORIGIN_CHECKPOINT
            and addendum["origin_journal"]["sha256"] == ORIGIN_JOURNAL
            and addendum["origin_journal"]["rows"] == 221, "Observation does not bind original105")
    validate_extended_catalog(repository, addendum["code"])
    effective = effective_bindings(base, addendum_sha, addendum["code"]["sha256"])
    require(approval == {"schema": SCHEMA + "_APPROVAL", "authorized": True,
        "basis": "USER_EXPLICITLY_APPROVED_VERSIONED_E5_TO_E10_OBSERVATION_AND_CONDITIONAL_CONTINUATION",
        "addendum_sha256": addendum_sha, "effective_bindings_digest": digest(effective),
        "execution_id": authorization["execution_id"], "output": str(RUN), "scope": SCOPE}, "Observation approval differs")
    for item in addendum["immutable_original_files"]:
        verify_file(item)
    verify_file(addendum["origin_journal"], prefix=True)
    verify_file(addendum["cpu_acceptance"])
    cpu = load(addendum["cpu_acceptance"]["path"])
    require(cpu["status"] == "PASS" and cpu["code_sha256"] == addendum["code"]["sha256"], "Observation CPU acceptance changed")
    return protocol, config, authorization, addendum, addendum_sha, base, effective


def _migration(root, journal, addendum, addendum_sha, base, bindings):
    from tools.exp17.formal.checkpoint import AtomicCheckpointStore, state_hash
    marker_path, active_path = ANCHOR / MARKER_NAME, ANCHOR / ACTIVE_NAME
    directory = Path(addendum["store_directory"])
    expected_claim = {"schema": SCHEMA + "_CLAIM", "addendum_sha256": addendum_sha,
                      "execution_id": addendum["origin_execution_id"], "output": str(RUN),
                      "bindings_sha256": digest(bindings)}
    fresh = not marker_path.exists()
    if fresh:
        require(len(journal.rows) == 221 and file_sha(journal.path) == ORIGIN_JOURNAL
                and not directory.exists() and not active_path.exists()
                and not any(r["event"] == "SOURCE_MIGRATION_COMMIT" for r in journal.rows),
                "Partial or conflicting observation migration")
        old_store = AtomicCheckpointStore(RUN / "recovery", base)
        old, old_receipt = old_store.load_latest(expected_sha256=ORIGIN_CHECKPOINT)
        validate_observation_boundary(old, journal, 105, 5, base)
        for item in addendum["legacy_caches"]:
            from tools.exp17.formal.sampler import SealedEpochCache
            SealedEpochCache.open(item["path"], bindings=base, expected_manifest_sha256=item["manifest_sha256"])
        derived = migrate_payload(old, base, bindings, addendum_sha)
        exclusive_json(marker_path, expected_claim)
        store = AtomicCheckpointStore(directory, bindings)
        receipt = store.save_latest(derived)
        checked, _ = store.load_latest(expected_sha256=receipt["sha256"])
        require(state_hash(checked) == state_hash(derived), "Derived105 strict reload differs")
        artifact = store.save_artifact("derived_update000105_observation.pt", derived, role="epoch_last")
        archived, _ = store.load_artifact("derived_update000105_observation.pt", expected_sha256=artifact["sha256"])
        require(state_hash(archived) == state_hash(derived), "Immutable derived105 differs")
        exclusive_json(Path(root) / "derived105_provenance.json", {"origin_checkpoint": old_receipt,
            "derived_checkpoint": receipt, "immutable_derived_checkpoint": artifact, "addendum_sha256": addendum_sha,
            "unchanged_training_fields": [k for k in old if k not in ("bindings", "extra")]})
        row = journal.append("SOURCE_MIGRATION_COMMIT", addendum_sha256=addendum_sha,
            bindings_sha256=digest(bindings), checkpoint_sha256=receipt["sha256"], origin_checkpoint_sha256=ORIGIN_CHECKPOINT,
            store_directory=str(directory.resolve()), best_directory=addendum["best_directory"],
            extension_root=addendum["extension_root"], completion_path=addendum["completion_path"],
            epoch10_gate_path=addendum["epoch10_gate_path"])
        exclusive_json(active_path, {"schema": SCHEMA + "_ACTIVE_STORE", "addendum_sha256": addendum_sha,
            "bindings_sha256": digest(bindings), "store_directory": str(directory.resolve()),
            "best_directory": addendum["best_directory"], "migration_event_sha256": row["sha256"],
            "migration_checkpoint_sha256": receipt["sha256"], "extension_root": addendum["extension_root"],
            "completion_path": addendum["completion_path"], "epoch10_gate_path": addendum["epoch10_gate_path"]})
    else:
        require(load(marker_path) == expected_claim and directory.is_dir() and active_path.is_file()
                and (Path(root) / "derived105_provenance.json").is_file(), "Partial observation migration requires review")
        store = AtomicCheckpointStore(directory, bindings)
    active = load(active_path)
    require(active["extension_root"] == addendum["extension_root"], "Active metadata root differs from Addendum")
    validate_active_store(active, journal, addendum_sha, bindings, directory)
    provenance = load(Path(root) / "derived105_provenance.json")
    immutable = provenance["immutable_derived_checkpoint"]
    require(immutable["file"] == "derived_update000105_observation.pt", "Unknown migrated105 artifact")
    archived, artifact_receipt = store.load_artifact(immutable["file"], expected_sha256=immutable["sha256"])
    require(artifact_receipt == immutable and archived["position"]["global_update"] == 105
            and provenance["addendum_sha256"] == addendum_sha, "Migrated105 lineage differs")
    payload, receipt = store.load_latest()
    last = next((r for r in reversed(journal.rows) if r["event"] in
                 ("UPDATE_COMMIT", "AUX_COMMIT", "SOURCE_MIGRATION_COMMIT")), None)
    require(last is not None and receipt["sha256"] == last["checkpoint_sha256"]
            and payload["position"]["global_update"] == journal.committed, "New active store and journal disagree")
    return store, payload


def run(repository, root):
    protocol, config, authorization, addendum, addendum_sha, base, bindings = validate(repository, root)
    root = Path(root).resolve()
    with RunLease(ANCHOR):
        journal = RunJournal(ANCHOR / "updates.jsonl")
        require(not any(r["event"] == "TRAIN_COMPLETE" for r in journal.rows), "Completed experiment cannot resume")
        store, recovered = _migration(root, journal, addendum, addendum_sha, base, bindings)
        from tools.exp17.formal.executor import run_session
        guard = lambda epoch, j, s: epoch_guard(epoch, j, s, root=root, addendum=addendum,
                                               addendum_sha=addendum_sha, bindings=bindings)
        legacy = lambda directory, manifest_sha: legacy_cache_bindings(directory, manifest_sha, addendum, base, bindings)
        adapter = None
        try:
            if journal.committed >= 210:
                decision = guard(10, journal, store)
                if decision is not None and decision["status"] == "R5_EPOCH10_HOLD":
                    exclusive_json(root / "completion.json", decision)
                    print(json.dumps(decision), flush=True)
                    return
            adapter = r5.adapter_type()(config, protocol, RUN, bindings)
            with best_namespace(RUN, bindings, addendum["best_directory"]):
                result = run_session(adapter, protocol, bindings, RUN, journal, store, recovered=recovered,
                                     legacy_cache_bindings=legacy, epoch_boundary=guard)
                if result["status"] == "R5_EPOCH10_PASS":
                    adapter.close()
                    recovered, _ = store.load_latest()
                    guard(10, journal, store)
                    adapter = r5.adapter_type()(config, protocol, RUN, bindings)
                    result = run_session(adapter, protocol, bindings, RUN, journal, store, recovered=recovered,
                                         legacy_cache_bindings=legacy, epoch_boundary=guard)
            exclusive_json(root / "completion.json", result)
            print(json.dumps(result), flush=True)
        except BaseException as error:
            journal.append("HARD_GATE", engineering_observation_addendum_sha256=addendum_sha, message=str(error))
            exclusive_json(root / f"hard_gate_{len(journal.rows):05d}.json", {"status": "OBSERVATION_HARD_GATE",
                "updates": journal.committed, "message": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            if adapter is not None:
                adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.action](REPOSITORY, args.root)


if __name__ == "__main__":
    main()
