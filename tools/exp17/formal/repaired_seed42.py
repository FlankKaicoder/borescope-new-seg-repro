"""Independent, plan-authorized Connected Statistics Seed42 experiment.

Prepare creates a new protocol and runtime configuration. Authorize requires
the frozen CPU/AMP receipts. Run retains the common budget/transaction engine.
An Epoch5 HOLD is terminal for this execution and never restarts evaluation.
"""
import argparse
import copy
import json
import shutil
import traceback
import uuid
from pathlib import Path

from tools.exp17.formal.authorization import (ACTIONS, RunJournal, RunLease, digest,
    exclusive_json, file_sha, load, protocol_bindings, require, source_catalog)

EXPERIMENT = "EXP17_J2_REPAIRED_SEED42"
ROOT = Path("/root/autodl-tmp/exp17_j2_repaired_seed42")
ANCHOR = Path("/root/.local/state/borescope") / EXPERIMENT
PREVIOUS_STEPS = Path("/root/autodl-tmp/exp17_j2_seed42_formal_v1/steps.jsonl")
PREVIOUS_SHA = "8d081cfef493c5c89378e1aa8ce4db414096c6ab597324b35dba58380a49b9f8"
G1_SHA = "613e4e154a872e0fd88791013f8e63716100470bb4a2f73084b702bb5cd37e44"
MODE = "replay_a_connected"


def val_scope(config, protocol_sha):
    from tools.exp17.formal.validation import MAX_PASSES, PASSES, ROLES, VAL_EPOCHS, VAL_FILES_SHA256
    return {"stage": "EXP17_R5_FROZEN_VAL_AUTHORIZATION", "authorized": True,
            "protocol_sha256": protocol_sha, "val_files_sha256": VAL_FILES_SHA256,
            "validation_epochs": list(VAL_EPOCHS), "roles": list(ROLES), "passes": list(PASSES),
            "maximum_native_passes": MAX_PASSES,
            **{key: config[key] for key in ("val_copy", "val_root")},
            "output": config["val_output"], "ledger_root": config["val_ledger_root"]}


def input_evidence(row):
    coverage = {k: v for k, v in row["coverage"].items()
                if k not in ("asset_cache", "N_images_decoded_this_call", "N_image_tensors_prepared_this_call")}
    return {"native": row["native_batch_fingerprint"], "coverage": coverage,
            "traces": [r["trace_sha256"] for r in row["explicit_augmentation_traces"]]}


def prepare(repository, gate):
    from tools.exp17.formal.r5_gate import RULE
    import importlib.util
    gate.mkdir(parents=True, exist_ok=False)
    original = repository / "results/exp17g_formal_protocol/formal_training_protocol_v1.json"
    require(file_sha(original) == G1_SHA, "G1 original changed")
    require(file_sha(PREVIOUS_STEPS) == PREVIOUS_SHA, "Original 2100-step evidence changed")
    evidence = [input_evidence(json.loads(line)) for line in PREVIOUS_STEPS.open()]
    require([r["native"]["step"] for r in evidence] == list(range(1, 2101)), "Original step coverage incomplete")
    exclusive_json(gate / "input_reference.json", evidence)
    protocol = load(original)
    protocol.update(experiment=EXPERIMENT, schema="EXP17_R5_CONNECTED_FORMAL_PROTOCOL_V1",
                    status="USER_APPROVED_R5_PLAN_PENDING_ENGINEERING_TESTS", date="2026-11-24")
    protocol["method"]["bn_mode"] = MODE
    protocol["engineering_repair"] = {"origin_protocol_sha256": G1_SHA,
        "origin_commit": "6d11a6e133192df181239422fcfc70e8f640b905", "gate_epoch": 5,
        "early_gate_rule": RULE, "production_derivative": "NATIVE_FORWARD_FIRST_ORDER_STATISTICS_VJP",
        "previous_steps_sha256": PREVIOUS_SHA,
        "input_reference_sha256": file_sha(gate / "input_reference.json"),
        "smoke_budget": {"A": 3, "N": 3, "component_autograd": 6, "joint_backward": 1,
                         "disposable_updates": 1, "VAL": 0}}
    protocol["authorization"] = {"basis": "USER_APPROVED_R5_PLAN_EXECUTION",
        "training_stage": "EXP17_R5_FORMAL_TRAIN_AUTHORIZATION", "val_stage": "EXP17_R5_FROZEN_VAL_AUTHORIZATION",
        "early_gate": "PLAN_PREAUTHORIZED_STRICT_CHECKPOINT_CONTINUATION_AT_105"}
    protocol["limitations"] = ["Connected production AMP memory must pass the separately budgeted smoke.",
        "Epoch5 early gate uses VAL for the predeclared continue decision.",
        "BN engineering threshold3 is uncalibrated against healthy Epoch5 states.",
        "Single seed cannot establish historical causality or cross-seed superiority."]
    framework = Path(importlib.util.find_spec("ultralytics").origin).parent
    protocol["framework_sources_sha256"]["nn/modules/conv.py"] = file_sha(framework / "nn/modules/conv.py")
    catalog = source_catalog(repository)
    protocol["code"]["files"], protocol["code"]["sha256"] = catalog, digest(catalog)
    exclusive_json(gate / "protocol.json", protocol)
    config = load("/root/autodl-tmp/exp17_j2_seed42_formal_gate_20261123/runtime_config.json")
    config.update(val_authorization=str(gate / "val_authorization.json"),
                  val_output=str(ROOT / "val"), val_ledger_root=str(ANCHOR / "val_passes"),
                  input_reference=str(gate / "input_reference.json"))
    exclusive_json(gate / "runtime_config.json", config)
    exclusive_json(gate / "val_authorization.json", val_scope(config, file_sha(gate / "protocol.json")))
    print(json.dumps({"status": "PREPARED_PENDING_CPU_AND_AMP", "gate": str(gate)}, flush=True))


def authorize(repository, gate):
    protocol = load(gate / "protocol.json")
    require(protocol["code"]["files"] == source_catalog(repository), "Source changed after candidate freeze")
    for name in ("cpu_tests.json", "smoke.json"):
        report = load(gate / name)
        require(report["status"] == "PASS" and report["code_sha256"] == protocol["code"]["sha256"],
                "R5 engineering prerequisite failed or used different sources")
    require(shutil.disk_usage(ROOT.parent).free >= 8 * 1024**3, "Less than approved 8GiB independent storage headroom")
    exclusive_json(gate / "formal_authorization.json", {
        "stage": "EXP17_R5_FORMAL_TRAIN_AUTHORIZATION", "gate": "USER_APPROVED_R5_PLAN_EXECUTION",
        "authorized": True, "experiment": EXPERIMENT, "authorized_actions": ACTIONS,
        "bindings": protocol_bindings(protocol, file_sha(gate / "protocol.json")),
        "runtime_config_sha256": file_sha(gate / "runtime_config.json"),
        "output": str(ROOT), "control_anchor": str(ANCHOR), "execution_id": uuid.uuid4().hex,
        "engineering_receipts": {n: file_sha(gate / n) for n in ("cpu_tests.json", "smoke.json")}})


def validate(repository, gate):
    protocol, config = load(gate / "protocol.json"), load(gate / "runtime_config.json")
    auth = load(gate / "formal_authorization.json")
    require(protocol["experiment"] == EXPERIMENT and protocol["method"]["bn_mode"] == MODE
            and protocol["formal_training_authorized"] is False and protocol["pending_fields"] == [], "Invalid R5 protocol")
    from tools.exp17.formal.r5_gate import RULE
    require(protocol["engineering_repair"]["early_gate_rule"] == RULE, "R5 early gate changed")
    require(protocol["code"]["files"] == source_catalog(repository)
            and digest(source_catalog(repository)) == protocol["code"]["sha256"], "R5 production source changed")
    require(auth["stage"] == "EXP17_R5_FORMAL_TRAIN_AUTHORIZATION"
            and auth["gate"] == "USER_APPROVED_R5_PLAN_EXECUTION" and auth["authorized"] is True
            and auth["experiment"] == EXPERIMENT and auth["authorized_actions"] == ACTIONS, "Invalid R5 training scope")
    bindings = protocol_bindings(protocol, file_sha(gate / "protocol.json"))
    require(auth["bindings"] == bindings and auth["runtime_config_sha256"] == file_sha(gate / "runtime_config.json"), "R5 identity conflict")
    require(auth["output"] == str(ROOT) and auth["control_anchor"] == str(ANCHOR) and len(auth["execution_id"]) == 32,
            "R5 output/anchor/execution identity conflict")
    require(load(gate / "val_authorization.json") == val_scope(config, bindings["protocol_sha256"]), "R5 VAL scope conflict")
    for name, expected in auth["engineering_receipts"].items():
        require(file_sha(gate / name) == expected and load(gate / name)["status"] == "PASS", "R5 receipt changed")
    require(file_sha(config["input_reference"]) == protocol["engineering_repair"]["input_reference_sha256"], "R5 input reference changed")
    return protocol, config, auth, bindings


def adapter_type():
    from tools.exp17.formal.native_adapter import NativeAdapter
    from tools.exp17.formal.validation import FrozenValAdapter
    class RepairedVal(FrozenValAdapter):
        authorization_stage = "EXP17_R5_FROZEN_VAL_AUTHORIZATION"
        write_pass_receipts = True
    class RepairedAdapter(NativeAdapter):
        bn_mode = MODE
        validator_class = RepairedVal

        def __init__(self, config, protocol, output, bindings):
            self.input_reference = load(config["input_reference"])
            super().__init__(config, protocol, output, bindings)

        def check_prepared(self, prepared, original, step):
            actual = input_evidence({"native_batch_fingerprint": original, "coverage": prepared.coverage,
                                     "explicit_augmentation_traces": [t.compact_record() for t in prepared.trace]})
            require(actual == self.input_reference[step - 1], f"Original J2 replay/Pair identity differs at step {step}")
    return RepairedAdapter


def epoch_boundary(epoch, journal, store):
    """Called before epoch6 scheduling or any attempt106, including on resume."""
    if epoch < 5:
        return None
    path = ROOT / "epoch005_gate.json"
    if not path.exists():
        require(epoch == 5 and journal.committed == 105 and journal.pending_aux is None,
                "Missing R5 gate outside the exact committed105 boundary")
        from tools.exp17.formal.r5_gate import decide
        payload, receipt = store.load_latest()
        require(payload["position"] == {"epoch": 4, "batch_position": 21, "global_update": 105}
                and payload["extra"]["epoch_events_complete"], "R5 gate checkpoint position differs")
        require(journal.rows[-1]["event"] == "AUX_COMMIT" and journal.rows[-1]["epoch"] == 5
                and journal.rows[-1]["checkpoint_sha256"] == receipt["sha256"], "R5 gate AUX not committed")
        reports = load(ROOT / "diagnostics_epoch005.json")
        periodic, _ = store.load_artifact("epoch005_full_recovery.pt")
        from tools.exp17.formal.checkpoint import state_hash
        require(state_hash(periodic) == state_hash(payload), "R5 periodic/Last states disagree")
        require(len(list((ANCHOR / "val_passes").glob("*.json"))) == 4, "R5 early pass count differs")
        from tools.exp14.formal_e0_e1 import hash_state
        for role, field in (("master", "state"), ("ema", "ema")):
            expected_state = hash_state({k: v for k, v in payload[field].items() if not k.startswith("region_projection.")})
            for mode in ("standard", "fixed"):
                directory = ROOT / "val" / f"epoch005_{role}_{mode}"
                receipt_file = load(directory / "pass_receipt.json")
                require(receipt_file["native_state_sha256"] == expected_state
                        and receipt_file["protocol_sha256"] == store.bindings["protocol_sha256"]
                        and (receipt_file["epoch"], receipt_file["role"], receipt_file["mode"]) == (5, role, mode)
                        and receipt_file["evaluation_sha256"] == file_sha(directory / "evaluation.json"),
                        "R5 early VAL pass completion/model identity differs")
        result = decide(reports, load(ROOT / "VAL_epoch005_master.json"))
        result.update(checkpoint=receipt, bindings=store.bindings, journal_prefix_sha256=file_sha(journal.path),
                      journal_last_sha256=journal.rows[-1]["sha256"], updates=105,
                      artifacts={p.name: file_sha(p) for p in [ROOT / "diagnostics_epoch005.json",
                          ROOT / "VAL_epoch005_master.json", ROOT / "VAL_epoch005_ema.json"]})
        exclusive_json(path, result)
        return {"status": "R5_EPOCH5_" + result["status"], "updates": 105}
    result = load(path)
    require(result["bindings"] == store.bindings, "R5 gate identity conflict")
    require(result["status"] == "PASS", "R5 Epoch5 HOLD is terminal; update106 forbidden")
    require(any(r["sha256"] == result["journal_last_sha256"] and r["event"] == "AUX_COMMIT" for r in journal.rows),
            "R5 gate is not in this journal lineage")
    for name, expected in result["artifacts"].items():
        require(file_sha(ROOT / name) == expected, "R5 accepted evidence changed")
    if journal.committed == 105:
        _, receipt = store.load_latest()
        require(receipt == result["checkpoint"], "R5 early continuation checkpoint differs")
    return None


def run(repository, gate):
    protocol, config, auth, bindings = validate(repository, gate)
    claim = {"experiment": EXPERIMENT, "authorization_sha256": digest(auth), "execution_id": auth["execution_id"],
             "output": str(ROOT), "cap": 2100}
    fresh = not ANCHOR.exists()
    if fresh:
        require(not ROOT.exists(), "Fresh R5 output already exists")
        ANCHOR.mkdir(parents=True, exist_ok=False)
        exclusive_json(ANCHOR / "CLAIM.json", claim)
        ROOT.mkdir(exist_ok=False)
        exclusive_json(ROOT / "EXECUTION_SEAL.json", claim)
    else:
        require(load(ANCHOR / "CLAIM.json") == load(ROOT / "EXECUTION_SEAL.json") == claim, "R5 permanent claim conflict")
    with RunLease(ANCHOR):
        journal = RunJournal(ANCHOR / "updates.jsonl", fresh=fresh)
        from tools.exp17.formal.checkpoint import AtomicCheckpointStore
        from tools.exp17.formal.executor import run_session
        store = AtomicCheckpointStore(ROOT / "recovery", bindings)
        recovered = None
        if not fresh:
            require(journal.committed == 105, "Only plan-approved105 continuation can resume automatically")
            epoch_boundary(5, journal, store)
            recovered, _ = store.load_latest()
        adapter = None
        try:
            adapter = adapter_type()(config, protocol, ROOT, bindings)
            result = run_session(adapter, protocol, bindings, ROOT, journal, store,
                                 recovered=recovered, epoch_boundary=epoch_boundary)
            if result["status"] == "R5_EPOCH5_PASS":
                adapter.close()
                recovered, _ = store.load_latest()
                epoch_boundary(5, journal, store)
                adapter = adapter_type()(config, protocol, ROOT, bindings)
                result = run_session(adapter, protocol, bindings, ROOT, journal, store,
                                     recovered=recovered, epoch_boundary=epoch_boundary)
            exclusive_json(ROOT / "completion.json", result)
            print(json.dumps(result), flush=True)
        except BaseException as error:
            evidence = {"status": "R5_HARD_GATE", "updates": journal.committed,
                        "message": str(error), "traceback": traceback.format_exc()}
            journal.append("HARD_GATE", message=str(error))
            exclusive_json(ROOT / f"hard_gate_{len(journal.rows):05d}.json", evidence)
            raise
        finally:
            if adapter is not None:
                adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "authorize", "run"))
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    {"prepare": prepare, "authorize": authorize, "run": run}[args.action](repository, args.gate)


if __name__ == "__main__":
    main()
