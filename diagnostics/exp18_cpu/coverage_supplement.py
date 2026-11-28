"""Supplement coverage from the sealed, independently accepted EXP18 metadata."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
import math
from pathlib import Path
import statistics
import tarfile

from tools.exp18.common import digest, require, sha256 as file_sha, write_json_exclusive

ARCHIVE = "0e4bd57cb5778f09f4de5c723e504199e0531b65a739f3711223385056b33463"
INVENTORY = "851da2f8027cfea1e1ba85e92cc834b40ac7bd1f029d74c50574d3822d4324eb"
STEPS = "3283f8b0c61945fe76080dc899ca467d0af5dd796ee64a6516d8942df0d3438d"
BINDING = "68aa155ea740173b58ea3a1e612f29c4828fb41f29d80b796697da5016b5c5ac"
CLASSES = ("Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion")
BUCKETS = ("1", "2_TO_4", "OVER_4")
PREFIX = "full_a10/replay/epochs/"


def entropy(counts):
    total = sum(counts)
    return -math.fsum(n / total * math.log2(n / total) for n in sorted(counts)) if total else None


def ratio(a, b):
    return a / b if b else None


def csv_write(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def records(bundle, index):
    archive, inventory = bundle / "core.tar.gz", bundle / "artifact_inventory.json"
    for path, expected in [(archive, ARCHIVE), (inventory, INVENTORY), (index, STEPS)]:
        require(path.is_file() and not path.is_symlink() and file_sha(path) == expected, "Accepted input SHA differs")
    doc = json.loads(inventory.read_text(encoding="utf-8"))
    require(doc["gates"]["bindings_sha256"] == BINDING, "Inventory binding differs")
    entries = {row["member"]: row for row in doc["entries"]}
    require(len(entries) == len(doc["entries"]), "Duplicate inventory member")
    with index.open(encoding="utf-8", newline="") as stream:
        expected_rows = list(csv.DictReader(stream))
    require(len(expected_rows) == 4200 and len({r["chunk"] for r in expected_rows}) == 200, "Accepted step index incomplete")
    summary_name = next(r["member"] for r in doc["gates"]["receipts"]
                        if r["member"].startswith("full_a10/replay/") and r["member"].endswith("/summary.json"))
    commits, summary = {}, None

    def member_bytes(stream, member):
        info = entries[member.name]
        require(member.isfile() and info["type"] == "file" and member.size == info["bytes"]
                and 0 < member.size < 5 * 1024**2, "Unsafe or oversized metadata member")
        raw = stream.extractfile(member).read()
        require(sha256(raw).hexdigest() == info["sha256"], "Inventory member SHA differs")
        return raw

    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name == summary_name:
                require(summary is None, "Duplicate ledger summary")
                summary = json.loads(member_bytes(stream, member))
            elif member.name.startswith(PREFIX) and member.name.endswith(".commit.json"):
                commit = json.loads(member_bytes(stream, member))
                epoch = commit["epoch"]
                require(member.name == PREFIX + f"epoch_{epoch:03d}.commit.json" and epoch not in commits,
                        "Duplicate or invalid epoch commit")
                commits[epoch] = commit
    require(summary is not None and summary["status"] == "CPU_DATA_READY_GPU_PENDING"
            and summary["completed_epochs"] == 200 and summary["bindings_sha256"] == BINDING
            and len(commits) == 200 and summary["commits"] == [commits[i] for i in range(200)],
            "Epoch commits differ from sealed summary")
    chunks = {PREFIX + item["chunk"]: item for item in commits.values()}
    require(len(chunks) == 200, "Repeated ledger chunk")
    completed = set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name not in chunks:
                continue
            c = chunks[member.name]
            ep = c["epoch"]
            require(ep not in completed and c["bindings_sha256"] == BINDING, "Duplicate epoch or unexpected binding")
            payload = member_bytes(stream, member)
            require(len(payload) == c["bytes"] and sha256(payload).hexdigest() == c["chunk_sha256"], "Chunk SHA changed")
            n = 0
            with gzip.GzipFile(fileobj=BytesIO(payload)) as epoch:
                for batch, raw in enumerate(epoch):
                    require(batch < 21, "Additional batch in ledger epoch")
                    pos = ep * 21 + batch
                    accepted, record = expected_rows[pos], json.loads(raw)
                    match = record["matching"]
                    require(record["native"]["step"] == int(accepted["step"]) == pos + 1
                            and int(accepted["epoch"]) == ep + 1 and int(accepted["batch_index"]) == batch
                            and match["epoch"] == ep and match["batch_index"] == batch
                            and digest(record) == accepted["record_sha256"] and match["sha256"] == accepted["matching_sha256"]
                            and digest({k: v for k, v in match.items() if k != "sha256"}) == match["sha256"]
                            and c["chunk"] == accepted["chunk"] and c["chunk_sha256"] == accepted["chunk_sha256"],
                            "Step differs from accepted record")
                    yield match, len(record["N_replay"])
                    n += 1
            require(n == 21, "Incomplete ledger epoch")
            completed.add(ep)
    require(len(completed) == 200, "Incomplete ledger corpus")


class Coverage:
    def __init__(self):
        self.tallies = defaultdict(Counter)
        self.polygons, self.groups = defaultdict(set), defaultdict(set)
        self.anchors, self.partners = {}, defaultdict(Counter)
        self.total = Counter()
        self.families = set()

    def add(self, matching):
        candidates, selected = matching["candidates"], matching["selected_candidates"]
        p = matching["permutation"]
        require(len({row["polygon_id"] for row in candidates}) == len(candidates)
                and len({row["polygon_id"] for row in selected}) == len(selected)
                and {row["polygon_id"] for row in selected} <= {row["polygon_id"] for row in candidates}
                and sorted(p) == list(range(len(selected))) and all(i != j for i, j in enumerate(p)),
                "Ledger does not contain a closed permutation over S")
        for name, rows in (("K", candidates), ("S", selected)):
            self.total[name] += len(rows)
            for row in rows:
                polygon, group, cls = row["polygon_id"], row["group_id"], row["class_id"]
                family = row["repair_version"]
                cells = row["p3_cells"]
                require(type(cls) is int and 0 <= cls < len(CLASSES) and type(cells) is int and cells > 0,
                        "Unexpected class or P3 count")
                self.families.add(family)
                bucket = BUCKETS[0 if cells == 1 else 1 if cells <= 4 else 2]
                prior = self.anchors.setdefault(polygon, (cls, group, row["source_id"], family))
                require(prior == (cls, group, row["source_id"], family), "Immutable Polygon identity changed")
                for key in ((cls, family, ""), (cls, "", bucket), ("", family, bucket), (cls, family, bucket)):
                    self.tallies[key][name] += 1
                    self.polygons[key, name].add(polygon)
                    self.groups[key, name].add(group)
                if name == "K":
                    self.tallies[polygon]["K"] += 1
        for i, row in enumerate(selected):
            self.partners[row["polygon_id"]][selected[p[i]]["polygon_id"]] += 1

    def outputs(self):
        anchors, edges, cross, summary = [], [], [], []
        for polygon, (cls, group, source, family) in sorted(self.anchors.items()):
            counts = self.partners[polygon]
            exposure = sum(counts.values())
            anchors.append({"arm": "a1_shuffled", "polygon_id": polygon, "class_id": cls, "class": CLASSES[cls],
                "group_id": group, "source_id": source, "repair_family": family, "K_exposures": self.tallies[polygon]["K"],
                "aux_exposures": exposure, "distinct_N_polygons": len(counts),
                "distinct_N_groups": len({self.anchors[n][1] for n in counts}),
                "max_partner_exposure_share": ratio(max(counts.values(), default=0), exposure),
                "partner_entropy_bits": entropy(counts.values()), "has_one_observed_partner": len(counts) == 1,
                "one_partner_single_exposure": len(counts) == 1 and exposure == 1,
                "one_partner_repeated_exposures": len(counts) == 1 and exposure > 1})
            edges.extend({"A_polygon_id": polygon, "N_polygon_id": n, "exposures": count} for n, count in sorted(counts.items()))
        for cls in (*range(len(CLASSES)), "ALL"):
            cohort = [r for r in anchors if cls == "ALL" or r["class_id"] == cls]
            active = [r for r in cohort if r["aux_exposures"]]
            single = [r for r in active if r["has_one_observed_partner"]]
            e, s = sum(r["aux_exposures"] for r in active), sum(r["aux_exposures"] for r in single)
            summary.append({"class_id": cls, "class": CLASSES[cls] if cls != "ALL" else "ALL",
                "K_unique_anchors": len(cohort), "S_unique_anchors": len(active), "zero_aux_anchors": len(cohort)-len(active),
                "aux_exposures": e, "single_partner_anchors": len(single),
                "single_partner_anchor_share_among_S": ratio(len(single), len(active)),
                "single_partner_exposures": s, "single_partner_exposure_share": ratio(s, e),
                "one_partner_single_exposure": sum(r["one_partner_single_exposure"] for r in single),
                "one_partner_repeated_exposures": sum(r["one_partner_repeated_exposures"] for r in single),
                "distinct_N_polygons_median": statistics.median(r["distinct_N_polygons"] for r in active) if active else None,
                "distinct_N_groups_median": statistics.median(r["distinct_N_groups"] for r in active) if active else None,
                "partner_entropy_bits_median": statistics.median(r["partner_entropy_bits"] for r in active) if active else None})
        families, classes = sorted(self.families), range(len(CLASSES))
        for granularity, cset, fset, bset in (
            ("class_version", classes, families, [""]), ("class_p3", classes, [""], BUCKETS),
            ("version_p3", [""], families, BUCKETS), ("class_version_p3", classes, families, BUCKETS)):
            totals = Counter()
            for key in product(cset, fset, bset):
                c, f, b = key
                row = {"granularity": granularity, "class_id": c, "class": CLASSES[c] if c != "" else "",
                       "repair_family": f, "P3_bucket": b}
                for pop in ("K", "S"):
                    e = self.tallies[key][pop]
                    row.update({pop+"_exposures": e, pop+"_unique_polygons": len(self.polygons[key, pop]),
                                pop+"_unique_groups": len(self.groups[key, pop])})
                    totals[pop] += e
                for suffix, rname in (("exposures", "exposure"), ("unique_polygons", "polygon"), ("unique_groups", "group")):
                    a, d = row["S_"+suffix], row["K_"+suffix]
                    require(a <= d, "S exceeds K in a cross cell")
                    row[rname+"_retention"] = ratio(a, d)
                cross.append(row)
            require(totals == self.total, "Cross block does not recover corpus totals")
        return anchors, edges, summary, cross


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = Coverage()
    slots = []
    for matching, n in records(args.archive_dir, args.steps):
        profile.add(matching)
        slots.append((matching["epoch"], n))
    a, edges, s, cross = profile.outputs()
    require(profile.total == {"K": 100867, "S": 22437}
            and len(a) == 521 and sum(r["aux_exposures"] > 0 for r in a) == 497 and len(cross) == 145,
            "Frozen coverage totals differ")
    args.output.mkdir(parents=True)
    for name, rows in (("partner_diversity_by_anchor", a), ("partner_edges", edges),
                       ("partner_diversity_summary_by_class", s), ("coverage_class_version_p3_cross", cross)):
        csv_write(args.output / (name + ".csv"), rows)
    epochs = Counter()
    for e, n in slots:
        epochs[e] += n
    write_json_exclusive(args.output / "coverage_receipt.json", {
        "status": "PASS_4200_SEALED_METADATA_STEPS", "archive_sha256": ARCHIVE, "inventory_sha256": INVENTORY,
        "steps_sha256": STEPS, "binding_sha256": BINDING, "K": profile.total["K"], "S": profile.total["S"],
        "anchor_rows": len(a), "edges": len(edges), "cross_rows": len(cross), "N_image_replays": sum(epochs.values()),
        "max_N_slots_single_epoch": max(epochs.values()),
        "max_N_slots_two_adjacent_epochs": max(epochs[e]+epochs[e+1] for e in range(199)),
        "max_N_slots_three_adjacent_epochs": max(epochs[e]+epochs[e+1]+epochs[e+2] for e in range(198)),
        "zero_budget": {"pixels": 0, "model": 0, "replay": 0, "VAL": 0, "TEST": 0},
        "csv_sha256": {f.name: file_sha(f) for f in sorted(args.output.glob("*.csv"))},
    })


if __name__ == "__main__":
    main()
