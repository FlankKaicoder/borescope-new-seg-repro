"""Stdlib-only verification of the sealed R5 Epoch5 HOLD evidence.

This reads JSON, Git blobs, archive members and checkpoint bytes. It never
imports torch, deserializes a checkpoint, reads images, or runs a model.
"""
import argparse
import hashlib
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path


ARCHIVE_SHA = "45310b0ed9c2df53b6d91cdade7cc1f94ba9422313141cd4e914b330ba0c24ce"
EXPERIMENT = "EXP17_J2_REPAIRED_SEED42"
PHYSICAL_KEYS = {"asset_cache", "N_images_decoded_this_call", "N_image_tensors_prepared_this_call"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name, condition):
        self.rows.append({"check": name, "pass": bool(condition)})


def git_blobs(repository, commit, paths):
    requests = b"".join(f"{commit}:{path}\n".encode() for path in paths)
    result = subprocess.run(["git", "cat-file", "--batch"], cwd=repository, input=requests,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    stream, blobs = io.BytesIO(result.stdout), {}
    for path in paths:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"Missing Git source blob {path}")
        data = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("Malformed Git batch boundary")
        blobs[path] = data
    return blobs


def verify(repository, evidence):
    c = Checks()
    run = evidence / "exp17_j2_repaired_seed42"
    gate_root = evidence / "exp17_r5_gate_20261124_v5"
    archive = evidence.parent / "full_private_archive.tar.gz"
    c.check("sealed_private_archive_sha256", sha(archive) == ARCHIVE_SHA)
    protocol = load(gate_root / "protocol.json")
    config = load(gate_root / "runtime_config.json")
    auth = load(gate_root / "formal_authorization.json")
    val_auth = load(gate_root / "val_authorization.json")
    early = load(run / "epoch005_gate.json")
    latest = load(run / "recovery/latest.json")
    claim, seal = load(run / "CLAIM.json"), load(run / "EXECUTION_SEAL.json")
    expected_bindings = {
        "protocol_sha256": sha(gate_root / "protocol.json"), "code_sha256": protocol["code"]["sha256"],
        "manifest_sha256": protocol["identity"]["trusted_manifest_sha256"],
        "native_identity_sha256": protocol["identity"]["native_train_identity_sha256"],
        "coco_sha256": protocol["identity"]["coco_sha256"], "seed": 42, "epochs": 100,
        "effective_updates": 2100, "environment": protocol["environment"]}
    c.check("protocol_experiment_and_connected_method", protocol["experiment"] == EXPERIMENT
            and protocol["method"] == {"lambda_supcon": 8, "temperature": .1, "lambda_pair": 4,
                "margin": .2, "bn_mode": "replay_a_connected", "checkpoint_prefix": False, "N_prefix": [0, 16]})
    c.check("training_authorization_bindings", auth["authorized"] is True and auth["bindings"] == expected_bindings
            and auth["experiment"] == EXPERIMENT and auth["stage"] == "EXP17_R5_FORMAL_TRAIN_AUTHORIZATION")
    c.check("runtime_config_byte_identity", auth["runtime_config_sha256"] == sha(gate_root / "runtime_config.json"))
    c.check("claim_and_seal_exact", claim == seal == {"experiment": EXPERIMENT, "authorization_sha256": digest(auth),
        "execution_id": auth["execution_id"], "output": auth["output"], "cap": 2100})
    c.check("early_gate_exact_bindings", early["bindings"] == expected_bindings)
    c.check("val_authorization_scope", val_auth["authorized"] is True
            and val_auth["stage"] == "EXP17_R5_FROZEN_VAL_AUTHORIZATION"
            and val_auth["protocol_sha256"] == expected_bindings["protocol_sha256"]
            and val_auth["validation_epochs"] == list(range(5, 101, 5))
            and val_auth["roles"] == ["master", "ema"] and val_auth["passes"] == ["standard", "fixed"]
            and val_auth["maximum_native_passes"] == 80 and val_auth["output"] == config["val_output"]
            and val_auth["ledger_root"] == config["val_ledger_root"]
            and val_auth["val_copy"] == config["val_copy"] and val_auth["val_root"] == config["val_root"])
    for name, expected in auth["engineering_receipts"].items():
        receipt = load(gate_root / name)
        c.check(f"engineering_receipt_{name}", sha(gate_root / name) == expected
                and receipt["status"] == "PASS" and receipt["code_sha256"] == protocol["code"]["sha256"])
    c.check("successful_smoke_exact_scope", load(gate_root / "smoke.json")["counts"]
            == protocol["engineering_repair"]["smoke_budget"])
    c.check("original_reference_byte_identity", sha(gate_root / "input_reference.json")
            == protocol["engineering_repair"]["input_reference_sha256"])
    reference = load(gate_root / "input_reference.json")
    steps = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    c.check("exact105_step_sequence", [r["step"] for r in steps] == list(range(1, 106)))
    c.check("105_images_raw_exposures", sum(r["images"] for r in steps) == 3340
            and sum(r["raw_targets"] for r in steps) == 6330)
    for row in steps:
        number, expected = row["step"], reference[row["step"] - 1]
        c.check(f"update_{number}_counters_and_epoch", row["step"] == row["effective_updates"] == row["optimizer_attempts"]
                and row["epoch"] == (number - 1) // 21 + 1 and row["batch_index"] == (number - 1) % 21)
        c.check(f"update_{number}_AMP_and_finite_state", row["amp_skipped"] is False and finite(row)
                and row["numeric_audit"]["overflow"] is False and row["numeric_audit"]["model_state_fp32_finite"] is True
                and row["numeric_audit"]["optimizer_state_finite"] is True
                and row["numeric_audit"]["native_step_called"] is True)
        c.check(f"update_{number}_BN_RNG_connected", row["BN_advances_per_A"] == 1 and row["N_extra_BN_updates"] == 0
                and row["native_RNG_unchanged_by_N"] is True and row["pair"]["bn_mode"] == "replay_a_connected"
                and row["pair"]["A_statistics_detached_on_N"] is False)
        c.check(f"update_{number}_native_trajectory", row["native_batch_fingerprint"] == expected["native"])
        c.check(f"update_{number}_trace_sha256", [r["trace_sha256"] for r in row["explicit_augmentation_traces"]]
                == expected["traces"])
        c.check(f"update_{number}_Pair_qualification", {k: v for k, v in row["coverage"].items()
                if k not in PHYSICAL_KEYS} == expected["coverage"])
    for epoch in range(1, 6):
        report = load(run / f"epoch{epoch:03d}.json")
        c.check(f"epoch_{epoch}_complete_tail_and_counts", report["complete"] is True and report["updates"] == 21
                and report["effective_updates"] == report["ema_updates"] == epoch * 21 and report["AMP_skips"] == 0
                and report["images"] == 668 and report["raw_instances"] == 1266
                and [r["images"] for r in steps[(epoch-1)*21:epoch*21]] == [32]*20+[28])

    raw_journal = (run / "updates.jsonl").read_bytes()
    lines = raw_journal.splitlines(keepends=True)
    journal = [json.loads(line) for line in lines]
    c.check("journal_221_complete_lines", len(journal) == 221 and all(line.endswith(b"\n") for line in lines))
    previous, committed, pending, aux = "0"*64, 0, None, None
    for index, row in enumerate(journal, 1):
        unsigned = {k: v for k, v in row.items() if k != "sha256"}
        c.check(f"journal_{index}_hash_chain", row["sequence"] == index and row["previous"] == previous
                and digest(unsigned) == row["sha256"])
        previous = row["sha256"]
        event, legal = row["event"], True
        if event == "ATTEMPT_BEGIN":
            legal = pending is None and aux is None and row["step"] == committed + 1
            pending = row["step"]
        elif event == "UPDATE_COMMIT":
            legal = pending == row["step"] and len(row["checkpoint_sha256"]) == 64
            committed, pending = row["step"], None
        elif event == "AUX_BEGIN":
            legal = aux is None and pending is None and row["epoch"] * 21 == committed
            aux = row["epoch"]
        elif event == "AUX_COMMIT":
            legal = aux == row["epoch"] and pending is None
            aux = None
        else:
            legal = event == "INITIAL_STATE_COMMIT" and index == 1
        c.check(f"journal_{index}_legal_transaction", legal)
    c.check("journal_closed105_AUX5_no106", committed == 105 and pending is None and aux is None
            and journal[-1]["event"] == "AUX_COMMIT" and journal[-1]["epoch"] == 5
            and not any(r.get("step", 0) >= 106 or r["event"] == "TRAIN_COMPLETE" for r in journal))
    c.check("early_gate_journal_prefix", early["journal_prefix_sha256"] == hashlib.sha256(raw_journal).hexdigest()
            and early["journal_last_sha256"] == journal[-1]["sha256"])
    c.check("latest_gate_AUX_pointer", latest == early["checkpoint"]
            and latest["sha256"] == journal[-1]["checkpoint_sha256"]
            and latest["position"] == {"epoch": 4, "batch_position": 21, "global_update": 105})
    for name, expected in early["artifacts"].items():
        c.check(f"early_gate_artifact_{name}", sha(run / name) == expected)

    checkpoint_receipts = {}
    for path in list((run / "recovery").glob("*.json")) + list((run / "best").glob("*.json")):
        receipt = load(path)
        checkpoint_receipts.setdefault(path.parent / receipt["file"], []).append((path.name, receipt))
    last_update = next(row for row in reversed(journal) if row["event"] == "UPDATE_COMMIT")
    slot1 = run / "recovery/recovery_slot_1.pt"
    checkpoint_receipts.setdefault(slot1, []).append(("UPDATE_COMMIT105", {"sha256": last_update["checkpoint_sha256"]}))
    all_pt = set(run.rglob("*.pt"))
    c.check("every_checkpoint_binary_has_receipt", all_pt == set(checkpoint_receipts) and len(all_pt) == 5)
    for path, receipts in checkpoint_receipts.items():
        actual = sha(path)
        for label, receipt in receipts:
            c.check(f"checkpoint_bytes_{path.name}_{label}", actual == receipt["sha256"]
                    and ("bytes" not in receipt or path.stat().st_size == receipt["bytes"]))
    periodic = load(run / "recovery/epoch005_full_recovery.pt.json")
    c.check("periodic_and_Last_same_bytes", periodic["sha256"] == latest["sha256"])
    bests = {role: load(run / f"best/best_{role}.json") for role in ("master", "ema")}
    for role, best in bests.items():
        c.check(f"best_{role}_first_selection", best["epoch"] == 5 and best["role"] == "best_"+role
                and best["fitness"] == 0 and best["strict_tensor_reload"] is True
                and best == load(run / f"best/best_{role}_epoch005.selection.json"))
    expected_names = {f"epoch005_{role}_{mode}.json" for role in ("master", "ema") for mode in ("standard", "fixed")}
    c.check("exact_four_VAL_claims", {p.name for p in (run / "val_passes").glob("*.json")} == expected_names)
    for role in ("master", "ema"):
        role_result = load(run / f"VAL_epoch005_{role}.json")
        for mode in ("standard", "fixed"):
            claim_name = f"epoch005_{role}_{mode}.json"
            pass_claim = load(run / "val_passes" / claim_name)
            directory = run / "val" / claim_name[:-5]
            receipt, evaluation = load(directory / "pass_receipt.json"), load(directory / "evaluation.json")
            c.check(f"VAL_{role}_{mode}_claim", pass_claim == {"epoch": 5, "role": role, "mode": mode,
                "protocol_sha256": expected_bindings["protocol_sha256"], "state": "ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY"})
            c.check(f"VAL_{role}_{mode}_model_and_result", receipt["epoch"] == 5 and receipt["role"] == role
                    and receipt["mode"] == mode and receipt["confidence"] == (.001 if mode == "standard" else .25)
                    and receipt["protocol_sha256"] == expected_bindings["protocol_sha256"]
                    and receipt["native_state_sha256"] == bests[role][f"{role}_native_state_sha256"]
                    and receipt["evaluation_sha256"] == sha(directory / "evaluation.json")
                    and receipt["result_digest"] == digest(role_result[mode]) and evaluation == role_result[mode])
            c.check(f"VAL_{role}_{mode}_frozen_counts_finite", evaluation["images"] == 154
                    and evaluation["instances"] == 296 and finite(evaluation))
        c.check(f"VAL_{role}_all_standard_zero", all(float(value) == 0 for value in role_result["standard"]["overall"].values()))
    probes = load(run / "diagnostics_epoch005.json")
    c.check("four_connected_TRAIN_probes", [p["batch_position"] for p in probes] == [1, 7, 11, 21]
            and all(p["bn_mode"] == "replay_a_connected" and p["optimizer_updates"] == 0 and finite(p) for p in probes))
    c.check("predeclared_scientific_HOLD", early["status"] == "HOLD" and early["updates"] == 105
            and early["master_mask_AP50_95"] == 0 and early["mismatched_layers"] == []
            and early["rule"] == protocol["engineering_repair"]["early_gate_rule"])
    c.check("terminal_completion_and_clean_exit", load(run / "completion.json") == {"status": "R5_EPOCH5_HOLD", "updates": 105}
            and load(gate_root / "formal_exit.json")["exit_code"] == 0 and not list(run.glob("hard_gate*")))
    c.check("E100_and_final_endpoints_absent", not list(run.glob("VAL_epoch100*"))
            and not (run / "final_endpoints.json").exists() and not (run / "final").exists())

    # Windows merges case-only directory names during extraction. Check anchor
    # artifacts against their exact Linux archive members to retain provenance.
    with tarfile.open(archive, "r:gz") as tar:
        anchored = ["CLAIM.json", "updates.jsonl"] + ["val_passes/"+name for name in sorted(expected_names)]
        for name in anchored:
            original = tar.extractfile(EXPERIMENT + "/" + name).read()
            c.check(f"exact_archive_anchor_member_{name}", original == (run / name).read_bytes())
        c.check("exact_archive_run_seal", tar.extractfile("exp17_j2_repaired_seed42/EXECUTION_SEAL.json").read()
                == (run / "EXECUTION_SEAL.json").read_bytes())

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    paths = sorted(protocol["code"]["files"])
    blobs = git_blobs(repository, commit, paths)
    actual_catalog = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    for name in paths:
        c.check("Git_blob_"+name, actual_catalog[name] == protocol["code"]["files"][name])
    listed = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "tools"], cwd=repository, text=True).splitlines()
    selected = {name for name in listed if name.endswith(".py") and "tests" not in Path(name).parts
                and not Path(name).name.startswith("test_")}
    c.check("Git_source_domain_exact", selected == set(paths))
    c.check("Git_source_catalog_digest", digest(actual_catalog) == protocol["code"]["sha256"])
    return {"status": "PASS" if all(r["pass"] for r in c.rows) else "FAIL",
        "verification_kind": "CODEX_OFFLINE_ENGINEERING_SELF_AUDIT",
        "checks": len(c.rows), "passed": sum(r["pass"] for r in c.rows),
        "failures": [r["check"] for r in c.rows if not r["pass"]], "details": c.rows,
        "Git_source_snapshot": commit, "source_catalog_sha256": protocol["code"]["sha256"],
        "private_archive_sha256": ARCHIVE_SHA, "protocol_sha256": expected_bindings["protocol_sha256"],
        "verified_effective_updates": 105, "verified_VAL_passes": 4,
        "checkpoint_binary_files": len(all_pt), "scientific_gate": "HOLD_ZERO_AP",
        "limits": ["Checkpoint byte hashes and JSON linkage verified without torch or tensor deserialization.",
            "Existing runtime Gate receipts report strict tensor reload; this script does not repeat it.",
            "Native model-state SHA links use recorded receipts and Best pointers, not recomputed tensor hashes.",
            "Windows case-only root merge is resolved against exact original archive member names.",
            "No images, model forward/backward, optimizer update or VAL/TEST were used by this verification.",
            "This is Codex engineering verification; ChatGPT independent scientific audit remains pending."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument("--evidence", type=Path, default=repository / "private_assets/exp17_r5/formal_e5_hold/full")
    parser.add_argument("--output", type=Path, default=repository / "results/exp17_j2_repaired_seed42/e5_integrity.json")
    args = parser.parse_args()
    report = verify(repository, args.evidence)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "checks", "passed", "failures", "Git_source_snapshot")}))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
