"""Stream committed CPU ledgers into aggregate coverage and matching audits.

Only JSON/CSV/gzip metadata is read.  Output directories and files are new and
exclusive; incomplete output and failed attempts are deliberately preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import statistics

from .common import digest, require, sha256, write_json_exclusive
from .pairing import CandidateExposure, MatchingConfig, build_matching, legal_pair, p3_bucket


CLASSES = ("Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion")
BUCKETS = ("1", "2_TO_4", "OVER_4")
FUNNEL_STAGES = ("K_SUPPORT_COMPLETE", "SAME_CLASS_CROSS_GROUP", "SAME_FAMILY", "AREA_RATIO_0_5_TO_2",
                 "SAME_P3_BUCKET", "CELLS_RATIO_0_5_TO_2", "S_MAXIMUM_CLOSED_PERMUTATION")


def _open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else Path(path).open(encoding="utf-8")


class _Population:
    def __init__(self):
        self.exposures = 0
        self.polygons, self.groups, self.sources = set(), set(), set()
        self.group_counts, self.group_weights = Counter(), Counter()
        self.weight = Fraction(0)
        self.b_availability = Counter()

    def add(self, row, weight=None):
        self.exposures += 1
        self.polygons.add(row["polygon_id"])
        self.groups.add(row["group_id"])
        if row.get("source_id") is not None:
            self.sources.add(row["source_id"])
        self.group_counts[row["group_id"]] += 1
        self.b_availability[{True: "has_B", False: "no_B", None: "unknown_B"}[row.get("has_real_crossgroup_B")]] += 1
        if weight is not None:
            self.weight += weight
            self.group_weights[row["group_id"]] += weight

    def record(self):
        counts = list(self.group_counts.values())
        weighted = list(self.group_weights.values())
        return {"exposures": self.exposures, "unique_polygons": len(self.polygons), "unique_groups": len(self.groups),
                "unique_sources": len(self.sources), "weight_sum": float(self.weight),
                "max_group_exposure_share": max(counts) / self.exposures if counts else None,
                "group_exposure_HHI": sum((n / self.exposures)**2 for n in counts) if counts else None,
                "max_group_weight_share": float(max(weighted) / self.weight) if weighted else None,
                "group_weight_HHI": float(sum((w / self.weight)**2 for w in weighted)) if weighted else None,
                "has_B_exposures": self.b_availability["has_B"], "no_B_exposures": self.b_availability["no_B"],
                "unknown_B_exposures": self.b_availability["unknown_B"]}


def _old_abn(record):
    """Reconstruct the original R5 population and original hierarchical weights."""
    explicit = record.get("abn_rows")
    if explicit is None:
        coverage = record.get("coverage", {})
        valid, native = coverage.get("valid_A_rows"), coverage.get("native_region_rows")
        if valid is None or native is None:
            return None
        native_map = {r.get("a_row", r.get("row_index")): r for r in native}
        require(len(native_map) == len(native) and len(valid) == len(set(valid)), "Ambiguous historical native/ABN rows")
        require(all(a in native_map for a in valid), "Historical ABN row has no native identity")
        explicit = [native_map[a] for a in valid]
        if "actual_valid_ABN" in coverage:
            require(coverage["actual_valid_ABN"] == len(explicit), "Historical ABN count mismatch")
    required = {"polygon_id", "group_id", "class_id", "source_id"}
    if any(not required.issubset(row) for row in explicit):
        return None
    require(len({r["polygon_id"] for r in explicit}) == len(explicit), "Repeated historical Polygon in one batch")
    class_groups = defaultdict(Counter)
    for row in explicit:
        require(type(row["class_id"]) is int and 0 <= row["class_id"] < 7, "Unknown historical class")
        class_groups[row["class_id"]][row["group_id"]] += 1
    result = {}
    for row in explicit:
        groups = class_groups[row["class_id"]]
        result[row["polygon_id"]] = (row, Fraction(1, len(class_groups) * len(groups) * groups[row["group_id"]]))
    return result


class _History:
    def __init__(self, path):
        self.path = None if path is None else Path(path)
        self.stream = None if path is None else _open_text(path)
        self.current = None
        self.last_read = 0
        self.source_sha256 = None if path is None else sha256(path)

    def for_step(self, step):
        if step > 2100:
            return None, "NOT_APPLICABLE_AFTER_R5_STEP_2100"
        if self.stream is None:
            return None, "UNKNOWN_NO_ORIGINAL_R5_LOG"
        while self.current is None or self.current["step"] < step:
            line = self.stream.readline()
            if not line:
                return None, "UNKNOWN_MISSING_HISTORICAL_STEP"
            self.current = json.loads(line)
            require(type(self.current.get("step")) is int and self.current["step"] > self.last_read,
                    "Historical step order changed")
            self.last_read = self.current["step"]
        if self.current["step"] != step:
            return None, "UNKNOWN_MISSING_HISTORICAL_STEP"
        rows = _old_abn(self.current)
        return rows, "AVAILABLE_ORIGINAL_R5_ABN" if rows is not None else "UNKNOWN_MISSING_ABN_FIELDS"

    def close(self):
        if self.stream is not None:
            self.stream.close()


def _audit_matching(record, verify_maximum):
    matching = record["matching"]
    require(matching.get("sha256") == digest({k: v for k, v in matching.items() if k != "sha256"}),
            "Recorded matching SHA differs")
    candidates = [CandidateExposure.from_mapping(r) for r in record["candidates"]["candidates"]]
    require([c.to_dict() for c in candidates] == matching["candidates"], "Candidate populations disagree")
    selected = [CandidateExposure.from_mapping(r) for r in matching["selected_candidates"]]
    a_rows = matching["a_rows"]
    require(a_rows == [c.a_row for c in selected] and len(a_rows) == len(set(a_rows)), "Selected A row identity differs")
    candidate_map = {c.a_row: c for c in candidates}
    require(len(candidate_map) == len(candidates) and all(candidate_map.get(c.a_row) == c for c in selected),
            "Selected rows are not a subset of the common candidate population")
    permutation, n_rows, weights = matching["permutation"], matching["n_rows"], matching["weights"]
    n = len(selected)
    require(sorted(permutation) == list(range(n)) and all(i != j for i, j in enumerate(permutation)),
            "Permutation is not a closed derangement of S")
    require(n_rows is not None and len(n_rows) == n and len(set(n_rows)) == n
            and all(type(v) is int and v >= 0 for v in n_rows), "Sparse N indices are unbound/repeated")
    require(matching.get("n_shuffled_indices") == [n_rows[j] for j in permutation], "Shuffled sparse N identities differ")
    require(len(weights) == n and all(math.isfinite(w) and w > 0 for w in weights), "Invalid auxiliary weights")
    config_data = matching["config"]
    config = MatchingConfig(matching_seed=config_data["matching_seed"], area_ratio_limit=config_data["area_ratio_limit"],
        cells_ratio_limit=config_data["cells_ratio_limit"], expected_scipy_version=config_data["expected_scipy_version"],
        cost_hash_bits=config_data["cost_hash_bits"])
    require(config.to_dict() == config_data and digest(config_data) == matching["config_sha256"], "Matching config binding differs")
    class_counts = Counter(c.class_id for c in selected)
    denominators = [len(class_counts) * class_counts[c.class_id] for c in selected]
    require(matching["weight_denominators"] == denominators and matching["weight_numerators"] == [1] * n,
            "Class/Polygon weight derivation differs")
    exact_weights = [Fraction(1, d) for d in denominators]
    require(all(w == float(expected) for w, expected in zip(weights, exact_weights)), "Loss weights changed")
    for i, j in enumerate(permutation):
        require(legal_pair(selected[i], selected[j], config), "Permutation used a forbidden edge")
        require(exact_weights[i] == exact_weights[j], "A/N weighted marginals differ")
        require(selected[i].has_real_crossgroup_B is not False, "Matched no-B row contradicts native population")
    require(sum(exact_weights, Fraction(0)) == (1 if n else 0), "Auxiliary weights do not sum to one")
    if verify_maximum:
        rebuilt = build_matching(candidates, config, epoch=matching["epoch"], batch_index=matching["batch_index"])
        rebuilt = rebuilt.bind_n_rows(dict(zip(a_rows, n_rows)))
        require(rebuilt.sha256 == matching["sha256"], "Independent maximum assignment/map recomputation differs")
    return candidates, selected, permutation, exact_weights


def _edge_funnel(candidates):
    stages = FUNNEL_STAGES[:-1]
    active = [set() for _ in stages]
    edges = [Counter() for _ in stages]
    active[0] = set(range(len(candidates)))
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i == j:
                continue
            conditions = (left.class_id == right.class_id and left.group_id != right.group_id,
                          left.repair_version == right.repair_version,
                          max(left.source_area_fraction, right.source_area_fraction)
                              <= 2 * min(left.source_area_fraction, right.source_area_fraction),
                          p3_bucket(left.p3_cells) == p3_bucket(right.p3_cells),
                          max(left.p3_cells, right.p3_cells) <= 2 * min(left.p3_cells, right.p3_cells))
            for index, condition in enumerate(conditions, 1):
                if not condition:
                    break
                active[index].add(i)
                edges[index][left.class_id] += 1
    result = []
    for index, stage in enumerate(stages):
        counts = Counter(candidates[i].class_id for i in active[index])
        for cls in range(7):
            result.append((cls, stage, counts[cls], edges[index][cls]))
    return result


def _csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or list(rows[0])
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return {"file": Path(path).name, "rows": len(rows), "sha256": sha256(path)}


def _quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_coverage(replay_root, output, historical_steps=None, *, verify_maximum=True):
    """Audit committed metadata and write aggregate public tables into a new dir.

    Missing historical ABN metadata is UNKNOWN; it is never inferred from N
    candidates.  Original R5 weights/intersections apply only to steps 1–2100.
    The second 100 epochs are explicitly outside that historical denominator.
    """
    replay_root, output = Path(replay_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = None
    try:
        history = _History(historical_steps)
        populations = defaultdict(_Population)
        strata = defaultdict(_Population)
        funnel = defaultdict(Counter)
        scalar_counts, historical_status = Counter(), Counter()
        partners, partner_groups, partner_exposures = defaultdict(set), defaultdict(set), Counter()
        sources, bindings_sha, expected_step, epoch_count = [], None, 1, 0
        for commit_path in sorted((replay_root / "epochs").glob("epoch_*.commit.json")):
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            require(commit["epoch"] == epoch_count, "Noncontiguous committed epoch domain")
            bindings_sha = bindings_sha or commit["bindings_sha256"]
            require(commit["bindings_sha256"] == bindings_sha, "Replay binding changed between epochs")
            name = commit["chunk"]
            require(isinstance(name, str) and Path(name).name == name and "\\" not in name, "Unsafe committed chunk name")
            chunk = replay_root / "epochs" / name
            require(not chunk.is_symlink() and chunk.is_file() and sha256(chunk) == commit["chunk_sha256"],
                    "Committed chunk bytes differ")
            source = {"file": chunk.name, "sha256": commit["chunk_sha256"], "bytes": chunk.stat().st_size,
                      "commit_file": commit_path.name, "commit_sha256": sha256(commit_path)}
            batches = 0
            with _open_text(chunk) as stream:
                for line in stream:
                    record = json.loads(line)
                    step = record["native"]["step"]
                    require(step == expected_step, "CPU ledger skipped/reordered a native batch")
                    require(record["matching"]["epoch"] == epoch_count
                            and record["matching"]["batch_index"] == batches, "Matching/native batch position differs")
                    candidates, selected, permutation, weights = _audit_matching(record, verify_maximum)
                    require(len(candidates) == record["coverage"]["candidate_counts"].get("support_complete_candidates", 0),
                            "Candidate funnel count differs from candidate population")
                    for population, rows in (("K", candidates), ("S", selected)):
                        for index, candidate in enumerate(rows):
                            row = candidate.to_dict()
                            weight = weights[index] if population == "S" else None
                            populations[(population, candidate.class_id)].add(row, weight)
                            strata[(population, candidate.class_id, candidate.repair_version, p3_bucket(candidate.p3_cells))].add(row, weight)
                    for trace in record.get("traces", []):
                        for row in trace["native_target_rows"]:
                            populations[("NATIVE_POST_AUGMENT_TARGETS", row["class_id"])].add(row)
                    for cls, stage, nodes, edges in _edge_funnel(candidates):
                        funnel[(cls, stage)].update(nodes=nodes, directed_edges=edges)
                    for cls, count in Counter(c.class_id for c in selected).items():
                        funnel[(cls, "S_MAXIMUM_CLOSED_PERMUTATION")].update(nodes=count)
                    for i, j in enumerate(permutation):
                        key = (selected[i].class_id, selected[i].polygon_id)
                        partners[key].add(selected[j].polygon_id)
                        partner_groups[key].add(selected[j].group_id)
                        partner_exposures[key] += 1
                    old, availability = history.for_step(step)
                    historical_status[availability] += 1
                    if old is not None:
                        require(record.get("historical_J1_R5_data_equal") is True,
                                "Historical ABN intersection requires exact historical data alignment")
                        candidate_ids = {c.polygon_id: c for c in candidates}
                        selected_ids = {c.polygon_id: c for c in selected}
                        for pid, (old_row, old_weight) in old.items():
                            cls = old_row["class_id"]
                            populations[("OLD_R5_ABN", cls)].add(old_row, old_weight)
                            for population, domain in (("K_INTERSECT_OLD_R5", candidate_ids),
                                                       ("S_INTERSECT_OLD_R5", selected_ids)):
                                if pid in domain:
                                    candidate = domain[pid]
                                    require((candidate.class_id, candidate.group_id, candidate.source_id)
                                            == (cls, old_row["group_id"], old_row["source_id"]),
                                            "Historical/new Polygon identity differs")
                                    populations[(population, cls)].add(old_row, old_weight)
                    scalar_counts.update({k: v for k, v in record["coverage"]["candidate_counts"].items()
                                          if type(v) is int})
                    scalar_counts.update(batches=1, matched_rows=len(selected), candidates=len(candidates),
                                         permutation_audits=1, weight_marginal_audits=1,
                                         maximum_assignment_recomputations=int(verify_maximum))
                    expected_step += 1
                    batches += 1
            require(batches == 21 and commit.get("counts", {}).get("batches", batches) == batches,
                    "Committed epoch is not exactly 21 native batches")
            require(sha256(chunk) == source["sha256"] and sha256(commit_path) == source["commit_sha256"],
                    "Committed metadata changed during audit")
            sources.append(source)
            epoch_count += 1
        require(epoch_count > 0, "No committed CPU epochs to summarize")
        has_history = historical_status["AVAILABLE_ORIGINAL_R5_ABN"] > 0
        unknown_history = sum(value for key, value in historical_status.items() if key.startswith("UNKNOWN"))
        history_availability = "PARTIAL_KNOWN_ROWS_ONLY" if has_history and unknown_history else ("AVAILABLE" if has_history else "UNKNOWN")
        tables = []
        population_rows = []
        for population in ("NATIVE_POST_AUGMENT_TARGETS", "K", "S", "OLD_R5_ABN", "K_INTERSECT_OLD_R5", "S_INTERSECT_OLD_R5"):
            for cls, name in enumerate(CLASSES):
                row = {"population": population, "class_id": cls, "class": name,
                       "scope": "ALL_AUDITED_STEPS" if population in ("NATIVE_POST_AUGMENT_TARGETS", "K", "S") else "AVAILABLE_R5_STEPS_1_2100_ONLY",
                       "availability": "AVAILABLE" if population in ("NATIVE_POST_AUGMENT_TARGETS", "K", "S") else history_availability,
                       **populations[(population, cls)].record()}
                if population in ("K", "NATIVE_POST_AUGMENT_TARGETS"):
                    row["weight_sum"] = "NOT_TRAINED_FULL_CANDIDATE_POPULATION"
                if row["availability"] == "UNKNOWN":
                    row.update({k: "UNKNOWN" for k in populations[(population, cls)].record()})
                population_rows.append(row)
        tables.append(_csv(output / "population_by_class.csv", population_rows))
        tables.append(_csv(output / "group_concentration.csv", population_rows))
        stratum_rows = []
        keys = sorted({(cls, family, bucket) for _, cls, family, bucket in strata})
        for cls, family, bucket in keys:
            k, s = strata[("K", cls, family, bucket)], strata[("S", cls, family, bucket)]
            stratum_rows.append({"class_id": cls, "class": CLASSES[cls], "repair_family": family,
                "P3_bucket": BUCKETS[bucket], "K_exposures": k.exposures, "S_exposures": s.exposures,
                "exposure_retention": s.exposures / k.exposures if k.exposures else None,
                "K_unique_polygons": len(k.polygons), "S_unique_polygons": len(s.polygons),
                "K_unique_groups": len(k.groups), "S_unique_groups": len(s.groups), "S_weight_sum": float(s.weight)})
        tables.append(_csv(output / "class_family_P3_coverage.csv", stratum_rows,
            ["class_id", "class", "repair_family", "P3_bucket", "K_exposures", "S_exposures", "exposure_retention",
             "K_unique_polygons", "S_unique_polygons", "K_unique_groups", "S_unique_groups", "S_weight_sum"]))
        funnel_rows = []
        for cls, name in enumerate(CLASSES):
            previous = None
            for stage in FUNNEL_STAGES:
                count = funnel[(cls, stage)]
                funnel_rows.append({"class_id": cls, "class": name, "stage": stage,
                    "anchor_exposures_with_partner": count["nodes"], "directed_edges": count["directed_edges"],
                    "dropped_exposures_from_previous_stage": None if previous is None else previous - count["nodes"]})
                require(previous is None or previous >= count["nodes"], "Constraint funnel unexpectedly grew")
                previous = count["nodes"]
        tables.append(_csv(output / "constraint_funnel_by_class.csv", funnel_rows))
        tables.append(_csv(output / "candidate_eligibility_funnel.csv",
            [{"received_population_counter": key, "exposures": value} for key, value in sorted(scalar_counts.items())]))
        partner_rows = []
        for cls, name in enumerate(CLASSES):
            keys = [key for key in partners if key[0] == cls]
            polygon_counts = [len(partners[key]) for key in keys]
            group_counts = [len(partner_groups[key]) for key in keys]
            partner_rows.append({"class_id": cls, "class": name, "matched_unique_polygons": len(keys),
                "matched_exposures": sum(partner_exposures[key] for key in keys),
                "distinct_partner_polygon_min": min(polygon_counts) if keys else None,
                "distinct_partner_polygon_median": statistics.median(polygon_counts) if keys else None,
                "distinct_partner_polygon_mean": statistics.fmean(polygon_counts) if keys else None,
                "distinct_partner_polygon_max": max(polygon_counts) if keys else None,
                "distinct_partner_group_median": statistics.median(group_counts) if keys else None,
                "distinct_partner_group_q25": _quantile(group_counts, .25),
                "distinct_partner_group_q75": _quantile(group_counts, .75)})
        tables.append(_csv(output / "partner_diversity_by_class.csv", partner_rows))
        retention = []
        for cls, name in enumerate(CLASSES):
            old = populations[("OLD_R5_ABN", cls)]
            k, s = populations[("K_INTERSECT_OLD_R5", cls)], populations[("S_INTERSECT_OLD_R5", cls)]
            retention.append({"class_id": cls, "class": name, "availability": history_availability,
                "scope": "AVAILABLE_R5_STEPS_1_2100_ONLY", "old_ABN_exposures": old.exposures if has_history else "UNKNOWN",
                "K_intersection_exposures": k.exposures if has_history else "UNKNOWN",
                "S_intersection_exposures": s.exposures if has_history else "UNKNOWN",
                "old_R5_weight_total": float(old.weight) if has_history else "UNKNOWN",
                "K_old_R5_weight_retention": float(k.weight / old.weight) if old.weight else (None if has_history else "UNKNOWN"),
                "S_old_R5_weight_retention": float(s.weight / old.weight) if old.weight else (None if has_history else "UNKNOWN")})
        tables.append(_csv(output / "original_R5_weight_retention.csv", retention))
        if historical_steps is not None:
            require(sha256(historical_steps) == history.source_sha256, "Historical R5 log changed during audit")
        summary = {"schema": "EXP18_CPU_COVERAGE_AUDIT_V1", "status": "CPU_COVERAGE_AUDIT_PASS_GPU_PENDING",
            "structural_gate": "PASS", "scientific_coverage_gate": "PENDING_REVIEW",
            "audited_epochs": epoch_count, "audited_batches": expected_step - 1, "bindings_sha256": bindings_sha,
            "counts": dict(scalar_counts), "historical_availability_batches": dict(historical_status),
            "historical_source_sha256": history.source_sha256, "tables": tables, "sources": sources,
            "maximum_assignment_recomputed": verify_maximum, "image_or_model_reads": 0,
            "historical_weight_definition": "mean_class(mean_A_group(mean_polygon)); only known R5 steps <=2100",
            "partial_history_is_not_zero": True,
            "interpretation": "K is all support-complete approved candidates; S is their common closed-permutation subset. No-B rows can enter K but cannot enter S under same-batch same-class cross-group matching."}
        write_json_exclusive(output / "summary.json", summary)
        text = ("# EXP18 CPU 覆盖审查\n\n"
                f"已审查 {epoch_count} 个已提交 epoch、{expected_step - 1} 个原生批次。闭合置换、合法边及逐行权重边际均通过。\n\n"
                f"K 候选曝光 {scalar_counts['candidates']:,} 次，S 实际匹配曝光 {scalar_counts['matched_rows']:,} 次。"
                "这两项均为重复训练曝光；唯一 Polygon 和来源组见逐类表。\n\n"
                "K 不要求 B；S 的每个异组同类伙伴本身也提供一个真实 B，因此 S 的 no-B 数必为零。"
                "面积与 P3 约束漏斗统计的是逐级具有合法伙伴的行；最终 S 还受闭合最大指派限制。\n\n"
                "原 R5 交集和旧层级权重仅在原记录可恢复且数据精确对齐的前 2,100 步计算。"
                "缺失记录明确标 UNKNOWN，后 2,100 步不属于历史分母。"
                "辅助权重按每批有效类等权、类内保留 Polygon 等权；组集中度单独报告。\n\n"
                "本报告只读元数据，不含图像、模型或真实 checkpoint。结构验收通过不代表覆盖足以支持科研主张；"
                "科学覆盖审议、GPU smoke、λ 和正式训练授权仍待后续 Gate。\n")
        with (output / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return summary
    except BaseException as error:
        write_json_exclusive(output / "failure.json", {"status": "HOLD", "error": repr(error),
                                                       "preserve_all_partial_artifacts": True})
        raise
    finally:
        if history is not None:
            history.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-steps", type=Path)
    args = parser.parse_args()
    summary = summarize_coverage(args.replay_root, args.output, args.historical_steps)
    print(json.dumps({"status": summary["status"], "batches": summary["audited_batches"]}))


if __name__ == "__main__":
    main()
