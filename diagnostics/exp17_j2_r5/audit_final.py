"""Snapshot audit of R5 C5→C6 evidence using JSON and file hashes only.

No model construction, tensor deserialization, image access or evaluation is
performed. Frozen-executor receipts supply the strict-reload evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

EXPERIMENT = "EXP17_J2_REPAIRED_SEED42"
C5 = "928f2726b9ca07bf44c1d1eee14beda25036c4e8957b5afd96f14a3fc0c17e07"
C6 = "6c5496624c5249d1c5c11327c63a04f2fc7a30d891001e1da3e922dccd62b7da"
PROTOCOL_SHA = "8aff5fa076e0fbf66d5473648f7f06c54d689f7b8ed2d7f2edfe8152d13a1c38"
CONFIG_SHA = "1d90672b5ae86d2d2c6a64bf7326bdfefe35ae473b86c6ef579ee66f67b9b763"
ADDENDUM_SHA = "c7e5bf6a49a505c192d1c649d7c25d1c95be229f6703a224bbca96449fe0f4e5"
ORIGIN_105 = "9193e57e2a31b7b7dafdfece107ddbce8f6f10a97a70390865cfcefcda5784ad"
ORIGIN_JOURNAL = "43c0acde2747c0d829b44786f03792a2e0455b5b8040d3809a0dfd6709076fb6"
EXTENSION = "diagnostics/exp17_j2_r5/observation_extension.py"
STORE, BEST = "recovery_e10_observation_v1", "best_e10_observation_v1"
ACTIVE = "R5_E10_OBSERVATION_ACTIVE_STORE.json"
EPOCHS = tuple(range(5, 101, 5))
ROLES, MODES = ("master", "ema"), ("standard", "fixed")
METRICS = tuple(f"metrics/{name}({kind})" for kind in ("B", "M")
                for name in ("precision", "recall", "mAP50", "mAP50-95"))
FOCUS = ("model.0.bn", "model.4.cv1.bn", "model.16.cv1.bn", "model.16.cv2.bn")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def binding_state_hash(value):
    """The checkpoint hash algorithm restricted to plain JSON metadata."""
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, list):
            result.update(b"list")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            result.update(repr(item).encode())
        else:
            raise ValueError("Non-JSON binding value")
    visit(value)
    return result.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def hex_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class Audit:
    def __init__(self, run, anchor, gate, observation):
        self.run, self.anchor, self.gate, self.observation = map(Path, (run, anchor, gate, observation))
        self.issues, self.counts, self.identities, self.files = [], {}, {}, {}
        self.checks = 0
        self.rows, self.roles, self.passes = [], {}, {}
        self.base, self.effective, self.protocol, self.addendum = None, None, None, None
        self.execution_id = None
        self.remote_roots = []

    def issue(self, code, label, kind="CONFLICT", **details):
        self.issues.append({"kind": kind, "code": code, "evidence": label, **details})

    def check(self, condition, code, label, kind="CONFLICT", **details):
        self.checks += 1
        if not condition:
            self.issue(code, label, kind, **details)
        return bool(condition)

    def json(self, path, label, *, required=True):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            if required:
                self.issue("MISSING_JSON", label, "MISSING")
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self.issue("UNREADABLE_JSON", label, message=type(error).__name__)
            return None
        self.files[label] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
        return value

    def blob(self, path, label, receipt):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            self.issue("MISSING_BOUND_BLOB", label, "MISSING")
            return
        actual_sha, size = file_sha(path), path.stat().st_size
        self.files[label] = {"sha256": actual_sha, "bytes": size}
        self.check(actual_sha == receipt.get("sha256") and size == receipt.get("bytes"), "BOUND_BLOB_SHA_OR_SIZE", label)

    def local_path(self, remote):
        remote = PurePosixPath(remote)
        for prefix, local in sorted(self.remote_roots, key=lambda item: len(str(item[0])), reverse=True):
            if remote.is_relative_to(prefix):
                relative = remote.relative_to(prefix)
                if ".." not in relative.parts:
                    return local / relative.as_posix()
        return None

    def identity(self):
        p = self.json(self.gate / "protocol.json", "gate/protocol.json")
        auth = self.json(self.gate / "formal_authorization.json", "gate/formal_authorization.json")
        config = self.json(self.gate / "runtime_config.json", "gate/runtime_config.json")
        val = self.json(self.gate / "val_authorization.json", "gate/val_authorization.json")
        seal = self.json(self.run / "EXECUTION_SEAL.json", "run/EXECUTION_SEAL.json")
        claim = self.json(self.anchor / "CLAIM.json", "anchor/CLAIM.json")
        if not all((p, auth, config, val, seal, claim)):
            return
        self.protocol = p
        self.check(file_sha(self.gate / "protocol.json") == PROTOCOL_SHA
                   and file_sha(self.gate / "runtime_config.json") == CONFIG_SHA,
                   "FROZEN_PROTOCOL_RUNTIME_BYTES", "gate/protocol_and_runtime.json")
        self.base = {"protocol_sha256": file_sha(self.gate / "protocol.json"), "code_sha256": p["code"]["sha256"],
            "manifest_sha256": p["identity"]["trusted_manifest_sha256"],
            "native_identity_sha256": p["identity"]["native_train_identity_sha256"], "coco_sha256": p["identity"]["coco_sha256"],
            "seed": 42, "epochs": 100, "effective_updates": 2100, "environment": p["environment"]}
        self.check(p["experiment"] == EXPERIMENT and p["formal_training_authorized"] is False and p["pending_fields"] == [], "PUBLIC_PROTOCOL_IDENTITY", "gate/protocol.json")
        expected_training = {"seed": 42, "epochs": 100, "batch": 32, "imgsz": 640, "amp": True,
            "optimizer": "AdamW", "gradient_clip_l2_max": 10., "drop_last": False, "workers": 2, "accumulate": 1}
        self.check(all(p["training"].get(key) == value for key, value in expected_training.items())
                   and p["effective_updates"] == 2100 and p["batches_per_epoch"] == 21, "FROZEN_TRAINING_PARAMETERS", "gate/protocol.json")
        self.check(p["method"] == {"N_prefix": [0, 16], "bn_mode": "replay_a_connected", "checkpoint_prefix": False,
            "lambda_pair": 4, "lambda_supcon": 8, "margin": .2, "temperature": .1}, "FROZEN_CONNECTED_METHOD", "gate/protocol.json")
        self.check(digest(p["code"]["files"]) == C5 == p["code"]["sha256"], "C5_CATALOG", "gate/protocol.json")
        self.check(auth.get("authorized") is True and auth.get("experiment") == EXPERIMENT
                   and auth.get("stage") == "EXP17_R5_FORMAL_TRAIN_AUTHORIZATION"
                   and auth.get("gate") == "USER_APPROVED_R5_PLAN_EXECUTION"
                   and auth.get("bindings") == self.base
                   and auth.get("runtime_config_sha256") == file_sha(self.gate / "runtime_config.json"),
                   "FORMAL_AUTHORIZATION_BINDINGS", "gate/formal_authorization.json")
        expected_claim = {"experiment": EXPERIMENT, "authorization_sha256": digest(auth), "execution_id": auth["execution_id"], "output": auth["output"], "cap": 2100}
        self.check(seal == claim == expected_claim, "PERMANENT_CLAIM_AND_EXECUTION_SEAL", "anchor/CLAIM.json")
        self.check(val.get("authorized") is True and val.get("stage") == "EXP17_R5_FROZEN_VAL_AUTHORIZATION"
                   and val.get("protocol_sha256") == self.base["protocol_sha256"]
                   and val.get("validation_epochs") == list(EPOCHS) and val.get("roles") == list(ROLES)
                   and val.get("passes") == list(MODES) and val.get("maximum_native_passes") == 80
                   and all(val.get(key) == config.get("val_" + key) for key in ("output", "ledger_root"))
                   and all(val.get(key) == config.get(key) for key in ("val_copy", "val_root"))
                   and val.get("val_files_sha256") == p["identity"]["VAL_files_manifest_sha256"],
                   "INDEPENDENT_VAL_AUTHORIZATION", "gate/val_authorization.json")
        self.check(config.get("manifest_sha256") == self.base["manifest_sha256"], "RUNTIME_MANIFEST_IDENTITY", "gate/runtime_config.json")
        for name, expected in auth["engineering_receipts"].items():
            receipt = self.json(self.gate / name, "gate/" + name)
            if receipt:
                self.check(file_sha(self.gate / name) == expected and receipt.get("status") == "PASS"
                           and receipt.get("code_sha256") == C5, "ENGINEERING_ACCEPTANCE_BINDING", "gate/" + name)
        self.remote_roots = [(PurePosixPath(auth["output"]), self.run), (PurePosixPath(auth["control_anchor"]), self.anchor),
            (PurePosixPath(config["val_authorization"]).parent, self.gate)]
        self.execution_id = auth["execution_id"]
        self.identities.update(protocol_sha256=self.base["protocol_sha256"], C5=C5,
            execution_id_sha256=digest(auth["execution_id"]), authorization_digest=digest(auth),
            manifest_sha256=self.base["manifest_sha256"], coco_sha256=self.base["coco_sha256"])

    def journal(self):
        path = self.anchor / "updates.jsonl"
        if not path.is_file():
            self.issue("MISSING_JOURNAL", "anchor/updates.jsonl", "MISSING")
            return
        committed, attempt, auxiliary, previous = 0, None, None, "0" * 64
        events = Counter()
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), 1):
            if not line.endswith("\n"):
                self.issue("TRUNCATED_JOURNAL_TAIL", "anchor/updates.jsonl", "MISSING")
                break
            row = json.loads(line)
            body = {key: value for key, value in row.items() if key != "sha256"}
            self.check(row.get("sequence") == index and row.get("previous") == previous and row.get("sha256") == digest(body), "JOURNAL_HASH_CHAIN", f"journal/record{index}")
            previous = row.get("sha256")
            event = row["event"]
            events[event] += 1
            if event == "ATTEMPT_BEGIN":
                self.check(attempt is None and auxiliary is None and row["step"] == committed + 1 and row["step"] <= 2100,
                           "OPTIMIZER_RESERVATION_ORDER", f"journal/record{index}")
                attempt = row["step"]
            elif event == "UPDATE_COMMIT":
                self.check(attempt == row["step"] and hex_sha(row.get("checkpoint_sha256")), "OPTIMIZER_COMMIT_ORDER", f"journal/record{index}")
                committed, attempt = row["step"], None
            elif event == "AUX_BEGIN":
                self.check(attempt is None and auxiliary is None and committed == row["epoch"] * 21, "AUX_BOUNDARY_ORDER", f"journal/record{index}")
                auxiliary = row["epoch"]
            elif event == "AUX_COMMIT":
                self.check(attempt is None and auxiliary == row["epoch"] and hex_sha(row.get("checkpoint_sha256")), "AUX_COMMIT_ORDER", f"journal/record{index}")
                auxiliary = None
            elif event == "SOURCE_MIGRATION_COMMIT":
                self.check(committed == 105 and attempt is None and auxiliary is None, "MIGRATION_BUDGET_BOUNDARY", f"journal/record{index}")
            elif event == "TRAIN_COMPLETE":
                self.check(committed == 2100 and auxiliary is None and attempt is None and row.get("updates") == 2100, "TRAIN_COMPLETE_BOUNDARY", f"journal/record{index}")
            self.rows.append(row)
        self.files["anchor/updates.jsonl"] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
        self.counts.update(journal_rows=len(self.rows), committed_updates=committed,
            attempts=events["ATTEMPT_BEGIN"], commits=events["UPDATE_COMMIT"], AUX_commits=events["AUX_COMMIT"],
            scheduled_VAL_AUX_commits=sum(row["event"] == "AUX_COMMIT" and row["epoch"] in EPOCHS for row in self.rows),
            source_migrations=events["SOURCE_MIGRATION_COMMIT"], TRAIN_COMPLETE_records=events["TRAIN_COMPLETE"],
            pending_attempt=attempt, pending_AUX=auxiliary, HARD_GATE_records=events["HARD_GATE"])
        self.check(events["INITIAL_STATE_COMMIT"] == 1, "SINGLE_FRESH_INITIAL_STATE", "anchor/updates.jsonl")
        self.check([r["epoch"] for r in self.rows if r["event"] == "AUX_COMMIT"]
                   == list(range(1, events["AUX_COMMIT"] + 1)), "CONTIGUOUS_AUX_EPOCHS", "anchor/updates.jsonl")
        for name, expected in (("committed_updates", 2100), ("attempts", 2100), ("commits", 2100),
                               ("AUX_commits", 100), ("scheduled_VAL_AUX_commits", 20), ("TRAIN_COMPLETE_records", 1), ("source_migrations", 1)):
            self.check(self.counts[name] == expected, "FINAL_BUDGET_INCOMPLETE", name, "INCOMPLETE", expected=expected, observed=self.counts[name])
        self.check(attempt is None and auxiliary is None, "PENDING_TRANSACTION", "anchor/updates.jsonl", "INCOMPLETE")

    def lineage(self):
        add = self.json(self.observation / "engineering_observation_addendum.json", "observation/addendum.json")
        approval = self.json(self.observation / "approval.json", "observation/approval.json")
        active = self.json(self.anchor / ACTIVE, "anchor/" + ACTIVE)
        provenance = self.json(self.observation / "derived105_provenance.json", "observation/derived105_provenance.json")
        catalog = self.json(self.observation / "production_source_catalog.json", "observation/production_source_catalog.json")
        if not self.base or not all((add, approval, active, provenance, catalog)):
            return
        self.addendum = add
        add_sha = file_sha(self.observation / "engineering_observation_addendum.json")
        self.check(add_sha == ADDENDUM_SHA, "FROZEN_OBSERVATION_ADDENDUM_BYTES", "observation/addendum.json")
        self.effective = {**self.base, "code_sha256": C6, "origin_code_sha256": C5,
            "engineering_observation_addendum_sha256": add_sha}
        self.remote_roots.append((PurePosixPath(add["extension_root"]), self.observation))
        self.check(add["origin_bindings"] == self.base and add["origin_checkpoint"]["sha256"] == ORIGIN_105
                   and add["origin_journal"]["sha256"] == ORIGIN_JOURNAL and add["origin_journal"]["rows"] == 221
                   and add.get("origin_execution_id") == self.execution_id
                   and add.get("origin_authorization_digest") == self.identities.get("authorization_digest"),
                   "ORIGINAL105_SNAPSHOT_BINDING", "observation/addendum.json")
        expected_files = {**self.protocol["code"]["files"], EXTENSION: add["code"]["files"].get(EXTENSION)}
        self.check(add["code"] == catalog and add["code"]["files"] == expected_files
                   and hex_sha(expected_files[EXTENSION]) and digest(expected_files) == C6 == add["code"]["sha256"],
                   "C5_TO_C6_EXACT_SOURCE_MIGRATION", "observation/production_source_catalog.json")
        self.check(approval.get("authorized") is True and approval.get("addendum_sha256") == add_sha
                   and approval.get("effective_bindings_digest") == digest(self.effective)
                   and approval.get("execution_id") == self.execution_id
                   and approval.get("scope") == add["scope"], "OBSERVATION_APPROVAL_BINDING", "observation/approval.json")
        expected_scope = {"origin_update": 105, "observation_epoch": 10, "observation_update": 210,
            "first_new_update": 106, "conditional_first_update": 211, "maximum_effective_updates": 2100, "maximum_native_VAL_passes": 80}
        self.check(all(add["scope"].get(k) == v for k, v in expected_scope.items()), "OBSERVATION_BUDGET_SCOPE", "observation/addendum.json")
        migrations = [row for row in self.rows if row["event"] == "SOURCE_MIGRATION_COMMIT"]
        fields = ("store_directory", "best_directory", "extension_root", "completion_path", "epoch10_gate_path")
        self.check(len(migrations) == 1 and migrations[0]["sha256"] == active.get("migration_event_sha256")
                   and migrations[0]["checkpoint_sha256"] == active.get("migration_checkpoint_sha256")
                   and migrations[0].get("addendum_sha256") == add_sha
                   and migrations[0].get("bindings_sha256") == digest(self.effective)
                   and migrations[0].get("origin_checkpoint_sha256") == ORIGIN_105
                   and active.get("addendum_sha256") == add_sha and active.get("bindings_sha256") == digest(self.effective)
                   and all(active.get(k) == add.get(k) == migrations[0].get(k) for k in fields),
                   "ACTIVE_STORE_JOURNAL_MIGRATION", "anchor/" + ACTIVE)
        journal = self.anchor / "updates.jsonl"
        if journal.is_file():
            with journal.open("rb") as stream:
                prefix = stream.read(add["origin_journal"]["bytes"])
            self.check(hashlib.sha256(prefix).hexdigest() == ORIGIN_JOURNAL and len(self.rows) >= 222
                       and self.rows[220]["sha256"] == add["origin_journal"]["last_row_sha256"]
                       and self.rows[220]["event"] == "AUX_COMMIT" and self.rows[220]["epoch"] == 5
                       and self.rows[220]["checkpoint_sha256"] == ORIGIN_105
                       and self.rows[221]["event"] == "SOURCE_MIGRATION_COMMIT", "ORIGINAL221_BYTE_PREFIX", "anchor/updates.jsonl")
        old_pointer = self.json(self.run / "recovery/latest.json", "run/recovery/latest.json")
        self.check(old_pointer == add["origin_latest_pointer"] == provenance["origin_checkpoint"], "ORIGINAL105_POINTER_PRESERVED", "run/recovery/latest.json")
        self.check(provenance.get("addendum_sha256") == add_sha
                   and provenance["derived_checkpoint"]["sha256"] == active["migration_checkpoint_sha256"], "DERIVED105_LINEAGE", "observation/derived105_provenance.json")
        for item in add["immutable_original_files"] + [add["cpu_acceptance"]]:
            local = self.local_path(item["path"])
            if local is None:
                self.issue("UNMAPPED_BOUND_EVIDENCE", PurePosixPath(item["path"]).name, "MISSING")
            else:
                self.blob(local, "immutable/" + PurePosixPath(item["path"]).name, item)
        immutable = provenance["immutable_derived_checkpoint"]
        receipt = self.json(self.run / STORE / (immutable["file"] + ".json"), "run/" + STORE + "/derived105_receipt.json")
        self.check(receipt == immutable and immutable.get("role") == "epoch_last"
                   and immutable.get("position") == {"epoch": 4, "batch_position": 21, "global_update": 105}
                   and immutable.get("bindings_sha256") == binding_state_hash(self.effective), "IMMUTABLE_DERIVED105_RECEIPT", "observation/derived105_provenance.json")
        self.blob(self.run / STORE / immutable["file"], "run/" + STORE + "/" + immutable["file"], immutable)
        latest = self.json(self.run / STORE / "latest.json", "run/" + STORE + "/latest.json")
        stateful = [row for row in self.rows if row["event"] in ("UPDATE_COMMIT", "AUX_COMMIT", "SOURCE_MIGRATION_COMMIT")]
        if latest and stateful:
            update = self.counts.get("committed_updates", 0)
            expected_position = {"epoch": (update - 1) // 21, "batch_position": (update - 1) % 21 + 1,
                                 "global_update": update}
            self.check(latest.get("sha256") == stateful[-1]["checkpoint_sha256"]
                       and latest.get("position") == expected_position and latest.get("role") == "last_recovery"
                       and latest.get("bindings_sha256") == binding_state_hash(self.effective), "ACTIVE_B1_LATEST_POINTER", "run/" + STORE + "/latest.json")
            self.blob(self.run / STORE / latest["file"], "run/" + STORE + "/" + latest["file"], latest)
        self.identities.update(C6=C6, engineering_observation_addendum_sha256=add_sha,
            effective_bindings_digest=digest(self.effective), effective_checkpoint_binding_hash=binding_state_hash(self.effective))

    def training(self):
        path = self.run / "steps.jsonl"
        if not path.is_file():
            self.issue("MISSING_STEPS", "run/steps.jsonl", "MISSING")
            return
        reference = self.json(self.gate / "input_reference.json", "gate/input_reference.json")
        if reference is not None and self.protocol:
            self.check(file_sha(self.gate / "input_reference.json") == self.protocol["engineering_repair"]["input_reference_sha256"], "FROZEN_INPUT_REFERENCE_SHA", "gate/input_reference.json")
        self.check(isinstance(reference, list) and len(reference) == 2100,
                   "2100_FROZEN_INPUT_REFERENCES_REQUIRED", "gate/input_reference.json", "MISSING" if reference is None else "CONFLICT")
        aggregates = defaultdict(lambda: {"count": 0, "images": 0, "instances": 0})
        count, skips, compared = 0, 0, 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    self.issue("TRUNCATED_STEPS_TAIL", "run/steps.jsonl", "MISSING")
                    break
                row = json.loads(line)
                count += 1
                label = f"steps/update{count}"
                self.check(row.get("step") == row.get("effective_updates") == row.get("optimizer_attempts") == count,
                           "STEP_COUNTERS", label)
                self.check(row.get("epoch") == (count - 1) // 21 + 1 and row.get("batch_index") == (count - 1) % 21
                           and row.get("images") == (28 if count % 21 == 0 else 32), "NATIVE_EPOCH_BATCH_AND_TAIL", label)
                self.check(all(finite(row.get(k)) for k in ("yolo_loss", "supcon_loss", "pair_loss", "total_loss", "gradient_norm", "clip_coefficient", "amp_scale")), "FINITE_NUMERICAL_RECORD", label)
                skips += row.get("amp_skipped") is not False
                self.check(row.get("BN_advances_per_A") == 1 and row.get("N_extra_BN_updates") == 0
                           and row.get("native_RNG_unchanged_by_N") is True, "BN_AND_RNG_INVARIANTS", label)
                if reference and count <= len(reference):
                    coverage = {k: v for k, v in row["coverage"].items() if k not in ("asset_cache", "N_images_decoded_this_call", "N_image_tensors_prepared_this_call")}
                    actual = {"native": row["native_batch_fingerprint"], "coverage": coverage,
                              "traces": [trace["trace_sha256"] for trace in row["explicit_augmentation_traces"]]}
                    self.check(actual == reference[count - 1], "FROZEN_REPLAY_AND_PAIR_ELIGIBILITY", label)
                    compared += 1
                epoch = row["epoch"]
                item = aggregates[epoch]
                item["count"] += 1
                item["images"] += row["images"]
                item["instances"] += row["raw_targets"]
                item["last_images"], item["last_step"] = row["images"], row["step"]
                item["BN_last"], item["parameter_last"] = row["bn_state_sha256"], row["parameter_state_sha256"]
        self.counts.update(step_rows=count, frozen_input_steps_checked=compared, AMP_skips=skips,
            original_image_exposures=sum(x["images"] for x in aggregates.values()), raw_instance_exposures=sum(x["instances"] for x in aggregates.values()))
        self.files["run/steps.jsonl"] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
        self.check(count == self.counts.get("committed_updates"), "METRICS_JOURNAL_AGREEMENT", "run/steps.jsonl", "INCOMPLETE")
        self.check(compared == count, "EVERY_STEP_HAS_FROZEN_INPUT_COMPARISON", "run/steps.jsonl", "MISSING")
        self.check(skips == 0, "AMP_SKIP_ZERO", "run/steps.jsonl")
        reports = []
        for file in sorted(self.run.glob("epoch[0-9][0-9][0-9].json")):
            value = self.json(file, "run/" + file.name)
            if not value:
                continue
            epoch = value["epoch"]
            item = aggregates.get(epoch, {})
            self.check(value.get("complete") is True and value.get("images") == item.get("images") == 668
                       and value.get("raw_instances") == item.get("instances") == 1266 and item.get("count") == 21
                       and item.get("last_images") == 28 and value.get("effective_updates") == value.get("ema_updates") == epoch * 21
                       and value.get("AMP_skips") == 0 and value.get("BN_last") == item.get("BN_last")
                       and value.get("parameter_state_last") == item.get("parameter_last"), "COMPLETE_EPOCH_EXPOSURE_AND_STATE", "run/" + file.name)
            reports.append(epoch)
        self.counts["epoch_reports"] = len(reports)
        self.check(reports == list(range(1, 101)), "100_EPOCH_REPORTS_REQUIRED", "run/epoch_reports", "INCOMPLETE")

    def validation(self):
        claim_paths = sorted((self.anchor / "val_passes").glob("*.json"))
        claims = {}
        for path in claim_paths:
            value = self.json(path, "anchor/val_passes/" + path.name)
            if not value:
                continue
            key = (value.get("epoch"), value.get("role"), value.get("mode"))
            self.check(key not in claims and key[0] in EPOCHS and key[1] in ROLES and key[2] in MODES
                       and path.name == f"epoch{key[0]:03d}_{key[1]}_{key[2]}.json"
                       and value.get("protocol_sha256") == (self.base or {}).get("protocol_sha256")
                       and value.get("state") == "ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY", "UNIQUE_AUTHORIZED_PASS_CLAIM", "anchor/val_passes/" + path.name)
            claims[key] = value
            directory = self.run / "val" / path.stem
            receipt = self.json(directory / "pass_receipt.json", "run/val/" + path.stem + "/pass_receipt.json")
            raw = self.json(directory / "evaluation.json", "run/val/" + path.stem + "/evaluation.json")
            if not receipt or not raw:
                continue
            self.check((receipt.get("epoch"), receipt.get("role"), receipt.get("mode")) == key
                       and receipt.get("protocol_sha256") == (self.base or {}).get("protocol_sha256")
                       and receipt.get("confidence") == (.001 if key[2] == "standard" else .25)
                       and receipt.get("evaluation_sha256") == file_sha(directory / "evaluation.json")
                       and receipt.get("result_digest") == digest(raw) and hex_sha(receipt.get("native_state_sha256")),
                       "PASS_COMPLETION_AND_MODEL_IDENTITY", "run/val/" + path.stem)
            self.passes[key] = (receipt, raw)
        expected = {(epoch, role, mode) for epoch in EPOCHS for role in ROLES for mode in MODES}
        self.counts.update(VAL_claims=len(claims), completed_pass_receipts=len(self.passes))
        output_stems = {path.parent.name for path in (self.run / "val").glob("*/evaluation.json")}
        receipt_stems = {path.parent.name for path in (self.run / "val").glob("*/pass_receipt.json")}
        claim_stems = {path.stem for path in claim_paths}
        self.check(output_stems == receipt_stems == claim_stems, "NO_UNCLAIMED_OR_INCOMPLETE_PASS_OUTPUTS", "run/val")
        self.check(set(claims) == expected and len(claim_paths) == 80, "80_UNIQUE_VAL_CLAIMS_REQUIRED", "anchor/val_passes", "INCOMPLETE")
        self.check(set(self.passes) == expected, "80_COMPLETE_NATIVE_PASSES_REQUIRED", "run/val", "INCOMPLETE")
        for path in sorted(self.run.glob("VAL_epoch[0-9][0-9][0-9]_*.json")):
            match = re.fullmatch(r"VAL_epoch(\d{3})_(master|ema)\.json", path.name)
            if not match:
                self.issue("UNKNOWN_VAL_ROLE_FILE", "run/" + path.name)
                continue
            epoch, role = int(match[1]), match[2]
            value = self.json(path, "run/" + path.name)
            if not value:
                continue
            self.check(epoch in EPOCHS and value.get("role") == role, "SCHEDULED_ROLE_SCOPE", "run/" + path.name)
            for mode in MODES:
                result = value.get(mode)
                pair = self.passes.get((epoch, role, mode))
                self.check(result is not None and pair is not None and digest(result) == digest(pair[1]), "ROLE_RESULT_PASS_LINK", "run/" + path.name, "MISSING" if pair is None else "CONFLICT")
                if result:
                    self.check(result.get("images") == 154 and result.get("instances") == 296
                               and all(finite(result.get("overall", {}).get(k)) and 0 <= result["overall"][k] <= 1 for k in METRICS), "FROZEN_VAL_COHORT_AND_METRICS", "run/" + path.name)
            standard = value["standard"]["overall"]
            fitness = standard["metrics/mAP50-95(M)"] + standard["metrics/mAP50-95(B)"]
            self.check(finite(value.get("fitness")) and math.isclose(value["fitness"], fitness, abs_tol=1e-12, rel_tol=0), "NATIVE_SELECTION_FITNESS", "run/" + path.name)
            classes = value["standard"].get("per_class", [])
            self.check([r.get("class_id") for r in classes] == list(range(7)) and sum(r.get("gt_instances", 0) for r in classes) == 296, "SEVEN_CLASS_VAL_RESULT", "run/" + path.name)
            self.check(all(finite(r.get(kind + "_" + metric)) and 0 <= r[kind + "_" + metric] <= 1
                           for r in classes for kind in ("box", "mask") for metric in ("ap50", "ap50_95", "precision", "recall")),
                       "FINITE_SEVEN_CLASS_METRICS", "run/" + path.name)
            fixed = value["fixed"].get("fixed_counts", {})
            per_class = fixed.get("per_class", [])
            self.check([r.get("class_id") for r in per_class] == list(range(7)), "SEVEN_CLASS_FIXED_RESULT", "run/" + path.name)
            for metric in ("gt", "detections", "box_tp", "box_fp", "box_fn", "mask_tp", "mask_fp", "mask_fn"):
                self.check(sum(r.get(metric, 0) for r in per_class) == fixed.get("overall", {}).get(metric), "FIXED_COUNTS_CLOSE", "run/" + path.name + "/" + metric)
            for row in [fixed.get("overall", {}), *per_class]:
                self.check(all(row.get(k + "_tp", 0) + row.get(k + "_fn", 0) == row.get("gt")
                               and row.get(k + "_tp", 0) + row.get(k + "_fp", 0) == row.get("detections") for k in ("box", "mask")), "FIXED_TP_FP_FN_CONSERVATION", "run/" + path.name)
            sizes = value["fixed"].get("size_recall", {})
            totals = sizes.get("overall", {})
            self.check(set(totals) == {"tiny", "small", "medium", "large"}
                       and sum(v.get("gt", 0) for v in totals.values()) == 296
                       and sum(v.get("mask_tp", 0) for v in totals.values()) == fixed.get("overall", {}).get("mask_tp"),
                       "FROZEN_SIZE_RECALL_COHORT", "run/" + path.name)
            for size, row in totals.items():
                self.check(row.get("mask_tp", 0) + row.get("mask_fn", 0) == row.get("gt")
                           and (row.get("mask_recall") is None if row.get("gt") == 0 else
                                finite(row.get("mask_recall")) and math.isclose(row["mask_recall"], row["mask_tp"] / row["gt"], abs_tol=1e-12)),
                           "SIZE_RECALL_COUNTS", "run/" + path.name + "/" + size)
            states = [self.passes[(epoch, role, mode)][0]["native_state_sha256"] for mode in MODES if (epoch, role, mode) in self.passes]
            self.check(len(states) == 2 and states[0] == states[1], "STANDARD_FIXED_SAME_ROLE_STATE", "run/" + path.name, "MISSING" if len(states) < 2 else "CONFLICT")
            self.roles[(epoch, role)] = value
        self.counts["VAL_role_results"] = len(self.roles)
        self.check(set(self.roles) == {(epoch, role) for epoch in EPOCHS for role in ROLES}, "40_ROLE_RESULTS_REQUIRED", "run/VAL_role_results", "INCOMPLETE")

    def diagnostics_and_gates(self):
        probes = 0
        events = []
        for path in sorted(self.run.glob("diagnostics_epoch[0-9][0-9][0-9].json")):
            events.append(int(path.stem[-3:]))
            reports = self.json(path, "run/" + path.name)
            self.check(isinstance(reports, list) and [r.get("batch_position") for r in reports] == [1, 7, 11, 21], "FOUR_FROZEN_TRAIN_PROBES", "run/" + path.name)
            if not isinstance(reports, list):
                continue
            for report in reports:
                self.check(report.get("optimizer_updates") == 0 and report.get("dynamic_lambda") is False
                           and report.get("bn_mode") == "replay_a_connected"
                           and all(finite(report.get("bn_mean_gap", {}).get(layer, {}).get("D")) and report["bn_mean_gap"][layer]["D"] >= 0 for layer in FOCUS),
                           "CONNECTED_ISOLATED_PROBE_AND_FINITE_BN", "run/" + path.name)
                probes += 1
        self.counts.update(TRAIN_diagnostic_probes=probes, diagnostic_events=len(list(self.run.glob("diagnostics_epoch[0-9][0-9][0-9].json"))))
        self.check(probes == 80, "80_FIXED_TRAIN_PROBES_REQUIRED", "run/diagnostics", "INCOMPLETE")
        self.check(events == list(EPOCHS), "20_SCHEDULED_TRAIN_PROBE_EVENTS_REQUIRED", "run/diagnostics", "INCOMPLETE")
        early = self.json(self.run / "epoch005_gate.json", "run/epoch005_gate.json")
        if early:
            self.check(early.get("status") == "HOLD" and early.get("updates") == 105 and early.get("mismatched_layers") == [], "HISTORICAL_EPOCH5_HOLD_PRESERVED", "run/epoch005_gate.json")
        gate = self.json(self.observation / "epoch010_gate.json", "observation/epoch010_gate.json")
        if not gate or not self.addendum:
            return
        self.check(gate.get("bindings") == self.effective and gate.get("updates") == 210
                   and gate.get("addendum_sha256") == self.identities.get("engineering_observation_addendum_sha256")
                   and gate.get("rule") == self.addendum.get("E10_rule"), "EPOCH10_BOUND_GATE", "observation/epoch010_gate.json")
        actual = self.roles.get((10, "master"))
        reports = self.json(self.run / "diagnostics_epoch010.json", "run/diagnostics_epoch010.json")
        if actual and isinstance(reports, list):
            rule = self.addendum["E10_rule"]
            bad = [layer for layer in rule["layers"] if sum(r["bn_mean_gap"][layer]["D"] >= rule["threshold"] for r in reports) >= rule["minimum_batches"]]
            ap = actual["standard"]["overall"]["metrics/mAP50-95(M)"]
            expected = "PASS" if ap > 0 and not bad else "HOLD"
            self.check(gate.get("status") == expected and gate.get("master_mask_AP50_95") == ap and gate.get("mismatched_layers") == bad, "EPOCH10_ORIGINAL_SCIENTIFIC_RULE", "observation/epoch010_gate.json")
            if self.counts.get("committed_updates", 0) > 210:
                self.check(expected == "PASS", "NO_UPDATE211_AFTER_HOLD", "observation/epoch010_gate.json")
        matching = [r for r in self.rows if r["event"] == "AUX_COMMIT" and r.get("epoch") == 10]
        self.check(len(matching) == 1 and matching[0]["sha256"] == gate.get("journal_last_sha256")
                   and matching[0]["checkpoint_sha256"] == gate.get("checkpoint", {}).get("sha256")
                   and gate.get("checkpoint", {}).get("bindings_sha256") == binding_state_hash(self.effective)
                   and gate.get("checkpoint", {}).get("position") == {"epoch": 9, "batch_position": 21, "global_update": 210},
                   "EPOCH10_GATE_CHECKPOINT_JOURNAL", "observation/epoch010_gate.json")
        if len(matching) == 1:
            lines = (self.anchor / "updates.jsonl").read_bytes().splitlines(keepends=True)
            prefix_sha = hashlib.sha256(b"".join(lines[:matching[0]["sequence"]])).hexdigest()
            self.check(gate.get("journal_prefix_sha256") == prefix_sha,
                       "EPOCH10_ACCEPTED_JOURNAL_BYTE_PREFIX", "observation/epoch010_gate.json")
        expected_artifacts = {str(PurePosixPath(self.addendum["output"]) / "diagnostics_epoch010.json")}
        for epoch in (5, 10):
            for role in ROLES:
                expected_artifacts.add(str(PurePosixPath(self.addendum["output"]) / f"VAL_epoch{epoch:03d}_{role}.json"))
                for mode in MODES:
                    stem = f"epoch{epoch:03d}_{role}_{mode}"
                    expected_artifacts.add(str(PurePosixPath(self.addendum["control_anchor"]) / "val_passes" / (stem + ".json")))
                    for name in ("pass_receipt.json", "evaluation.json"):
                        expected_artifacts.add(str(PurePosixPath(self.addendum["output"]) / "val" / stem / name))
        self.check(set(gate.get("artifacts", {})) == expected_artifacts, "COMPLETE_ACCEPTED_E10_GATE_ARTIFACT_SET", "observation/epoch010_gate.json")
        for remote, expected in gate.get("artifacts", {}).items():
            local = self.local_path(remote)
            if local is None or not local.is_file():
                self.issue("MISSING_GATE_EVIDENCE", PurePosixPath(remote).name, "MISSING")
            else:
                self.check(file_sha(local) == expected, "ACCEPTED_EPOCH10_ARTIFACT_SHA", "gate-artifact/" + local.name)

    def models_and_selection(self):
        if not self.base or not self.effective:
            return
        periodic = []
        for area, binding in (("recovery", self.base), (STORE, self.effective)):
            for path in sorted((self.run / area).glob("epoch*_full_recovery.pt.json")):
                match = re.fullmatch(r"epoch(\d{3})_full_recovery\.pt\.json", path.name)
                value = self.json(path, "run/" + area + "/" + path.name)
                if not match or not value:
                    continue
                epoch = int(match[1])
                periodic.append(epoch)
                self.check(epoch in EPOCHS and (epoch == 5) == (area == "recovery")
                           and value.get("position") == {"epoch": epoch - 1, "batch_position": 21, "global_update": epoch * 21}
                           and value.get("role") == "periodic" and value.get("bindings_sha256") == binding_state_hash(binding),
                           "PERIODIC_EPOCH_AND_BINDING", "run/" + area + "/" + path.name)
                self.blob(path.with_suffix(""), "run/" + area + "/" + value["file"], value)
        self.counts["periodic_recoveries"] = len(periodic)
        self.check(sorted(periodic) == list(EPOCHS), "20_PERIODIC_SNAPSHOTS_REQUIRED", "run/periodic", "INCOMPLETE")
        for role in ROLES:
            candidates = sorted((epoch, record) for (epoch, selected_role), record in self.roles.items() if selected_role == role)
            best, selected_epochs = None, []
            for epoch, record in candidates:
                if best is None or record["fitness"] >= best[1]["fitness"]:
                    best = epoch, record
                    selected_epochs.append(epoch)
            histories = []
            for area, binding in (("best", self.base), (BEST, self.effective)):
                for path in sorted((self.run / area).glob(f"best_{role}_epoch*.selection.json")):
                    value = self.json(path, "run/" + area + "/" + path.name)
                    if not value:
                        continue
                    epoch = value["epoch"]
                    histories.append(epoch)
                    self.check(value.get("role") == "best_" + role and value.get("bindings_sha256") == binding_state_hash(binding)
                               and value.get("strict_tensor_reload") is True and value.get("optimizer_included") is False
                               and value.get("historical_selection_record_is_weight_archive") is False, "BEST_HISTORY_FORMAT_AND_BINDING", "run/" + area + "/" + path.name)
                    self.best_identity(value, epoch, role, path.name)
            self.check(sorted(histories) == selected_epochs, "LATEST_EQUAL_FITNESS_SELECTION_HISTORY", "best-history/" + role,
                       "INCOMPLETE" if len(candidates) < 20 else "CONFLICT", expected=selected_epochs, observed=histories)
            area = BEST if any(epoch >= 10 for epoch in selected_epochs) else "best"
            pointer = self.json(self.run / area / f"best_{role}.json", "run/" + area + f"/best_{role}.json")
            if best and pointer:
                self.check(pointer.get("epoch") == best[0] and math.isclose(pointer.get("fitness", -1), best[1]["fitness"], abs_tol=1e-12, rel_tol=0), "CURRENT_BEST_SELECTION", "best-pointer/" + role)
                self.best_identity(pointer, best[0], role, "best-pointer/" + role)
                self.blob(self.run / area / pointer["file"], "run/" + area + "/" + pointer["file"], pointer)
            self.counts["Best_" + role + "_selected_epochs"] = histories
        endpoints = self.json(self.run / "final_endpoints.json", "run/final_endpoints.json")
        if not endpoints:
            return
        for role, filename in (("master", "epoch100_last_master_fp32.pt"), ("ema", "epoch100_ema_fp32.pt")):
            receipt = self.json(self.run / "final" / (filename + ".json"), "run/final/" + filename + ".json")
            pair = self.passes.get((100, role, "standard"))
            if receipt:
                self.check(receipt == endpoints.get(role) and receipt.get("file") == filename and receipt.get("role") == "epoch100_" + role
                           and receipt.get("epoch") == 100 and receipt.get("effective_updates") == 2100
                           and receipt.get("strict_native_state_reload") is True and receipt.get("native_ultralytics_loader_checked") is True
                           and receipt.get("primary_evaluation_input") is (role == "master")
                           and receipt.get("bindings_sha256") == binding_state_hash(self.effective), "FINAL_NATIVE_EXPORT_RECEIPT", "run/final/" + filename + ".json")
                self.check(pair is not None and receipt.get("native_state_sha256") == pair[0]["native_state_sha256"], "FINAL_LAST_EMA_VAL_IDENTITY", "run/final/" + filename + ".json", "MISSING" if pair is None else "CONFLICT")
                self.blob(self.run / "final" / filename, "run/final/" + filename, receipt)
        complete = [row for row in self.rows if row["event"] == "TRAIN_COMPLETE"]
        if len(complete) == 1:
            self.check(complete[0].get("endpoints") == endpoints, "TRAIN_COMPLETE_ENDPOINT_IDENTITY", "anchor/updates.jsonl")

    def best_identity(self, value, epoch, role, label):
        record = self.roles.get((epoch, role))
        self.check(record is not None and math.isclose(value.get("fitness", -1), record["fitness"], abs_tol=1e-12, rel_tol=0), "BEST_FITNESS_EVALUATION_LINK", label, "MISSING" if record is None else "CONFLICT")
        for stored_role in ROLES:
            pair = self.passes.get((epoch, stored_role, "standard"))
            self.check(pair is not None and value.get(stored_role + "_native_state_sha256") == pair[0]["native_state_sha256"], "BEST_CANONICAL_NATIVE_STATES", label, "MISSING" if pair is None else "CONFLICT")

    def completion(self):
        historical = self.json(self.run / "completion.json", "run/completion.json")
        if historical:
            self.check(historical.get("status") == "R5_EPOCH5_HOLD" and historical.get("updates") == 105, "ORIGINAL_COMPLETION_HISTORICAL", "run/completion.json")
        current = self.json(self.observation / "completion.json", "observation/completion.json")
        if current:
            self.check(current.get("status") == "FORMAL_100_EPOCH_COMPLETE_PENDING_INDEPENDENT_RESULT_GATE"
                       and current.get("updates") == 2100, "ACTIVE_COMPLETION_100_EPOCH", "observation/completion.json", "INCOMPLETE")
            endpoints = self.json(self.run / "final_endpoints.json", "run/final_endpoints.json")
            if endpoints:
                self.check(current.get("endpoints") == endpoints, "COMPLETION_FINAL_ENDPOINTS", "observation/completion.json")
            self.check(set(current.get("best", {})) == set(ROLES), "COMPLETION_HAS_BOTH_BEST_ROLES", "observation/completion.json")
            for role in ROLES:
                best = current.get("best", {}).get(role)
                if best:
                    self.best_identity(best.get("checkpoint", {}), best["epoch"], role, "completion/Best_" + role)
                    record = self.roles.get((best["epoch"], role))
                    area = BEST if best["epoch"] >= 10 else "best"
                    pointer = self.json(self.run / area / f"best_{role}.json", "run/" + area + f"/best_{role}.json")
                    self.check(best.get("role") == role and best.get("checkpoint") == pointer
                               and best.get("evaluation") == record and best.get("rule") == "LATEST_EQUAL_NATIVE_FITNESS"
                               and record is not None and best.get("mask_mAP50_95") == record["standard"]["overall"]["metrics/mAP50-95(M)"],
                               "COMPLETION_BEST_CURRENT_POINTER_AND_EVALUATION", "completion/Best_" + role)
        windows = []
        for root, filename, label in ((self.gate, "formal_exit.json", "gate/formal_exit.json"),
                                       (self.observation, "exit.json", "observation/exit.json")):
            value = self.json(root / filename, label)
            if value:
                self.check(value.get("exit_code") == 0 and value.get("automatic_restart") is False
                           and finite(value.get("started_unix")) and finite(value.get("ended_unix"))
                           and value["ended_unix"] >= value["started_unix"], "COMPLETED_PROCESS_RECEIPT", label, "INCOMPLETE")
                if finite(value.get("started_unix")) and finite(value.get("ended_unix")):
                    windows.append(value["ended_unix"] - value["started_unix"])
        self.counts["recorded_process_seconds"] = sum(windows)

    def execute(self):
        for section in (self.identity, self.journal, self.lineage, self.training, self.validation,
                        self.diagnostics_and_gates, self.models_and_selection, self.completion):
            try:
                section()
            except (KeyError, TypeError, ValueError, OSError) as error:
                self.issue("UNSUPPORTED_OR_INCOMPLETE_SECTION_FORMAT", section.__name__, "MISSING",
                           error_type=type(error).__name__)
        conflicts = sum(issue["kind"] == "CONFLICT" for issue in self.issues)
        missing = len(self.issues) - conflicts
        budget_complete = self.counts.get("committed_updates") == 2100 and self.counts.get("epoch_reports") == 100
        status = "PASS_FINAL_EVIDENCE_COMPLETE" if not self.issues else "FAIL_EVIDENCE_CONFLICT" if budget_complete and conflicts else "INCOMPLETE"
        return {"schema": "EXP17_R5_C5_C6_FINAL_OFFLINE_AUDIT_V1", "status": status, "experiment": EXPERIMENT,
            "checks": self.checks, "conflicts": conflicts, "incomplete_or_missing": missing, "counts": self.counts,
            "identities": self.identities, "issues": self.issues, "evidence_files": self.files,
            "scope": {"JSON_and_file_SHA_only": True, "checkpoint_tensor_loads": 0, "model_imports": 0,
                "model_forwards": 0, "new_VAL": 0, "TEST_access": 0,
                "strict_tensor_reload_evidence": "FROZEN_EXECUTOR_RECEIPTS_AND_BOUND_C5_C6_SOURCE",
                "historical_rotated_Best_blobs_required": False,
                "privacy": "ONLY_SANITIZED_RELATIVE_EVIDENCE_LABELS_AND_IDENTITY_DIGESTS; NO_EXECUTION_ID_OR_SOURCE_GROUPS",
                "independent_scientific_audit": "PENDING"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("run-root", "anchor-root", "gate-root", "observation-root", "output"):
        parser.add_argument("--" + argument, required=True, type=Path)
    args = parser.parse_args()
    report = Audit(args.run_root, args.anchor_root, args.gate_root, args.observation_root).execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "checks", "conflicts", "incomplete_or_missing", "counts")}))
    raise SystemExit(0 if report["status"] == "PASS_FINAL_EVIDENCE_COMPLETE" else
                     2 if report["status"] == "INCOMPLETE" else 1)


if __name__ == "__main__":
    main()
