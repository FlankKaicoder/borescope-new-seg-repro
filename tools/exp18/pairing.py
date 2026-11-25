"""Deterministic, closed True/Shuffled matching on TRAIN-only CPU metadata.

The square assignment includes a self edge meaning 'discard this row'.
Removing self edges leaves closed cycles on the same A and N population;
an arbitrary intersection of unmatched bipartite domains would not do so.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from collections.abc import Mapping


SCHEMA = "EXP18_CLOSED_CPU_MATCHING_V1"
_HASH_FIELDS = ("n_sha256", "augmentation_sha256", "mask_sha256", "manifest_record_sha256",
                "source_sha256", "matrix_sha256", "source_valid_mask_sha256", "n_valid_mask_sha256")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _field(value, *names):
    for name in names:
        if name in value:
            return value[name]
    raise ValueError("Missing metadata field: " + "/".join(names))


@dataclass(frozen=True)
class CandidateExposure:
    a_row: int
    a_slot: int
    source_id: str
    group_id: str
    polygon_id: str
    class_id: int
    repair_version: str
    n_sha256: str
    source_area_fraction: float
    p3_cells: int
    augmentation_sha256: str
    mask_sha256: str
    manifest_record_sha256: str = ""
    source_sha256: str = ""
    matrix_sha256: str = ""
    source_valid_mask_sha256: str = ""
    n_valid_mask_sha256: str = ""
    annotation_ordinal: int | None = None
    native_post_format_ordinal: int | None = None
    n_row: int | None = None
    has_real_crossgroup_B: bool | None = None

    def __post_init__(self):
        _require(all(type(x) is int and x >= 0 for x in (self.a_row, self.a_slot)),
                 "A pool row and native image slot must be nonnegative integers")
        _require(type(self.class_id) is int and 0 <= self.class_id < 7, "Invalid frozen seven-class identity")
        _require(type(self.p3_cells) is int and self.p3_cells > 0, "Candidate requires actual positive native P3 cells")
        _require(all(isinstance(x, str) and x and x == x.strip()
                     for x in (self.source_id, self.group_id, self.polygon_id)), "Missing source/group/Polygon identity")
        _require(self.repair_version in ("V1", "V2", "V3", "V4"), "Unknown repair version family; do not infer it")
        _require(isinstance(self.source_area_fraction, (float, int)) and not isinstance(self.source_area_fraction, bool)
                 and math.isfinite(self.source_area_fraction) and 0 < self.source_area_fraction <= 1,
                 "Native polygon frame fraction must be finite and positive")
        for name in _HASH_FIELDS:
            value = getattr(self, name)
            _require(_is_sha(value) or (name not in _HASH_FIELDS[:3] and value == ""), f"Invalid {name}")
        for name in ("annotation_ordinal", "native_post_format_ordinal", "n_row"):
            value = getattr(self, name)
            _require(value is None or (type(value) is int and value >= 0), f"Invalid {name}")
        _require(self.has_real_crossgroup_B is None or type(self.has_real_crossgroup_B) is bool,
                 "B availability must be known bool or absent")

    @classmethod
    def from_mapping(cls, row):
        values = dict(row)
        for name, aliases in {"a_row": ("row_index",), "a_slot": ("slot",),
                              "source_area_fraction": ("area_frame_fraction",)}.items():
            if name not in values:
                values[name] = _field(values, *aliases)
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})

    def to_dict(self):
        return asdict(self)

    @property
    def identity_sha256(self):
        return _digest(self.to_dict())


@dataclass(frozen=True)
class CandidateCollection:
    candidates: tuple[CandidateExposure, ...]
    counts: dict
    exclusions: tuple[dict, ...]

    def to_dict(self):
        return {"candidates": [x.to_dict() for x in self.candidates], "counts": dict(self.counts),
                "exclusions": list(self.exclusions),
                "population": "INPUT_NATIVE_ROWS_TO_ALL_SUPPORT_COMPLETE_521_CANDIDATES_NO_B_REQUIREMENT"}


def collect_candidates(native_rows, trusted_by_polygon, support_by_row, native_geometry_by_polygon):
    """Collect all support-complete approved rows, independently of B existence.

    Rows may include zero-cell targets; counts describe the received population,
    not raw annotations filtered upstream.  An absent N is a normal exclusion;
    unknown/mismatched support or trusted identity is an error, never guessed.
    """
    rows = [dict(row) for row in native_rows]
    a_rows = [_field(row, "row_index", "a_row") for row in rows]
    _require(len(a_rows) == len(set(a_rows)), "Duplicate native A pool row")
    counts = Counter(input_native_rows=len(rows))
    candidates, exclusions = [], []
    for row, a_row in zip(rows, a_rows):
        pid = row["polygon_id"]
        trusted = trusted_by_polygon.get(pid)
        if trusted is None:
            counts["no_admitted_N"] += 1
            exclusions.append({"a_row": a_row, "polygon_id": pid, "reason": "NO_ADMITTED_N"})
            continue
        counts["admitted_521_rows"] += 1
        if "record_sha256" in trusted:
            _require(trusted["record_sha256"] == _digest({k: v for k, v in trusted.items() if k != "record_sha256"}),
                     "Trusted manifest record changed after loading")
        if "split" in trusted:
            _require(trusted["split"] == "train", "Non-TRAIN record cannot enter the matching candidates")
        if "asset_type" in trusted:
            _require(trusted["asset_type"] == "ALIGNED", "Only admitted ALIGNED N assets are eligible")
        _require((row["source_id"], row["group_id"], row["class_id"])
                 == (trusted["source_id"], trusted["group_id"], trusted["class_id"]), "Trusted/native identity differs")
        if "polygon_id" in trusted:
            _require(trusted["polygon_id"] == pid, "Trusted Polygon map key differs")
        cells = row["p3_cells"]
        _require(type(cells) is int and cells >= 0, "Unknown native P3 cell count")
        if not cells:
            counts["admitted_native_raster_missing"] += 1
            exclusions.append({"a_row": a_row, "polygon_id": pid, "reason": "ZERO_NATIVE_P3_CELLS"})
            continue
        counts["admitted_poolable_rows"] += 1
        has_b = any(other["class_id"] == row["class_id"] and other["group_id"] != row["group_id"]
                    and other["p3_cells"] > 0 for other in rows)
        counts["poolable_has_B" if has_b else "poolable_no_B"] += 1
        _require(a_row in support_by_row, "Missing augmented Valid support; do not assume it is complete")
        support = support_by_row[a_row]
        _require(support["polygon_id"] == pid and type(support["support_complete"]) is bool,
                 "Support identity or completeness differs")
        if "p3_cells" in support:
            _require(support["p3_cells"] == cells, "Support and native P3 cell counts differ")
        if not support["support_complete"]:
            counts["incomplete_augmented_valid_support"] += 1
            exclusions.append({"a_row": a_row, "polygon_id": pid, "reason": "INCOMPLETE_A_N_VALID_SUPPORT"})
            continue
        geometry = native_geometry_by_polygon[pid]
        if isinstance(geometry, Mapping):
            if "source_area_fraction" in geometry or "area_frame_fraction" in geometry:
                area = _field(geometry, "source_area_fraction", "area_frame_fraction")
            else:
                area = geometry["source_area_continuous"] / (geometry["native_width"] * geometry["native_height"])
        else:
            area = geometry
        optional = {"manifest_record_sha256": "record_sha256", "source_sha256": "a_sha256",
                    "matrix_sha256": "matrix_identity_sha256", "source_valid_mask_sha256": "source_valid_mask_sha256",
                    "n_valid_mask_sha256": "n_valid_mask_sha256"}
        candidate = CandidateExposure(
            a_row=a_row, a_slot=_field(row, "slot", "a_slot"), source_id=row["source_id"], group_id=row["group_id"],
            polygon_id=pid, class_id=row["class_id"], repair_version=trusted["repair_version"],
            n_sha256=trusted["n_sha256"], source_area_fraction=float(area), p3_cells=cells,
            augmentation_sha256=_field(row, "augmentation_sha256", "trace_sha256"),
            mask_sha256=support["mask_sha256"],
            annotation_ordinal=row.get("annotation_ordinal", row.get("native_label_ordinal")),
            native_post_format_ordinal=row.get("native_post_format_ordinal"),
            has_real_crossgroup_B=has_b, **{name: trusted.get(source, "") for name, source in optional.items()})
        candidates.append(candidate)
        counts["support_complete_candidates"] += 1
        counts["candidate_has_B" if has_b else "candidate_no_B"] += 1
    counts["candidate_unique_polygons"] = len({c.polygon_id for c in candidates})
    counts["candidate_unique_groups"] = len({c.group_id for c in candidates})
    return CandidateCollection(tuple(sorted(candidates, key=lambda c: c.a_row)), dict(counts), tuple(exclusions))


@dataclass(frozen=True)
class MatchingConfig:
    matching_seed: int = 180042
    area_ratio_limit: float = 2.
    cells_ratio_limit: float = 2.
    expected_scipy_version: str | None = None
    cost_hash_bits: int = 16

    def __post_init__(self):
        _require(type(self.matching_seed) is int and self.matching_seed >= 0, "Invalid independent matching seed")
        _require(self.area_ratio_limit == self.cells_ratio_limit == 2., "EXP18 v1 fixes area/cells ratios to [0.5,2]")
        _require(self.cost_hash_bits == 16, "EXP18 v1 fixes the integer SHA secondary cost to 16 bits")
        _require(self.expected_scipy_version is None or
                 (isinstance(self.expected_scipy_version, str) and self.expected_scipy_version), "Invalid SciPy version pin")

    @classmethod
    def from_protocol(cls, protocol):
        _require(isinstance(protocol, Mapping) and isinstance(protocol.get("matching"), Mapping),
                 "Protocol must supply the frozen matching object")
        matching = dict(protocol["matching"])
        fixed = {"same_batch": True, "same_class": True, "different_group": True,
                 "same_repair_family": True, "area_field": "native_polygon_area_divided_by_frame_area",
                 "area_ratio_max": 2., "p3_buckets": [1, 4], "cells_ratio_max": 2.,
                 "closed_permutation": True, "no_fixed_points": True, "unique_N": True,
                 "objective": "maximum_non_diagonal_assignment_then_seeded_integer_cost",
                 "permutation_after_N_pooling": True}
        _require(not (set(matching) - set(fixed) - {"seed", "expected_scipy_version"}),
                 "Unknown matching protocol field")
        for key, expected in fixed.items():
            actual = matching.get(key)
            _require(actual == expected and (not isinstance(expected, bool) or type(actual) is bool),
                     f"Frozen matching protocol differs: {key}")
        _require("seed" in matching, "Matching protocol must explicitly bind its seed")
        return cls(matching_seed=matching["seed"], area_ratio_limit=matching["area_ratio_max"],
                   cells_ratio_limit=matching["cells_ratio_max"],
                   expected_scipy_version=matching.get("expected_scipy_version"))

    def to_dict(self):
        return {**asdict(self), "p3_cell_buckets": ["1", "2_TO_4", "OVER_4"],
                "area_field": "FROZEN_NATIVE_FLOAT32_YOLO_POLYGON_FRAME_FRACTION",
                "reduction": "MEAN_ACTIVE_CLASS_MEAN_RETAINED_POLYGON",
                "same_class": True, "same_repair_version_family": True, "different_group": True}


def p3_bucket(cells):
    _require(type(cells) is int and cells > 0, "Expected actual positive native P3 cells")
    return 0 if cells == 1 else 1 if cells <= 4 else 2


def legal_pair(left, right, config=None):
    config = config or MatchingConfig()
    return (left.class_id == right.class_id and left.repair_version == right.repair_version
            and left.group_id != right.group_id and p3_bucket(left.p3_cells) == p3_bucket(right.p3_cells)
            and max(left.source_area_fraction, right.source_area_fraction)
                <= config.area_ratio_limit * min(left.source_area_fraction, right.source_area_fraction)
            and max(left.p3_cells, right.p3_cells) <= config.cells_ratio_limit * min(left.p3_cells, right.p3_cells))


@dataclass(frozen=True)
class MatchingPlan:
    candidates: tuple[CandidateExposure, ...]
    selected_candidates: tuple[CandidateExposure, ...]
    permutation: tuple[int, ...]
    weights: tuple[float, ...]
    weight_denominators: tuple[int, ...]
    n_rows: tuple[int, ...] | None
    exclusions: tuple[dict, ...]
    config: MatchingConfig
    epoch: int
    batch_index: int
    solver_metadata: dict

    @property
    def a_rows(self):
        return tuple(c.a_row for c in self.selected_candidates)

    @property
    def a_indices(self):
        return self.a_rows

    @property
    def n_indices(self):
        return self.n_rows

    @property
    def n_true_indices(self):
        return self.n_rows

    @property
    def n_shuffled_indices(self):
        return None if self.n_rows is None else tuple(self.n_rows[j] for j in self.permutation)

    @property
    def selected_a_slots(self):
        return tuple(sorted({c.a_slot for c in self.selected_candidates}))

    def bind_n_rows(self, a_to_n_row):
        _require(isinstance(a_to_n_row, Mapping), "N row binding requires an explicit A-to-sparse-N row map")
        _require(all(c.a_row in a_to_n_row for c in self.selected_candidates), "Sparse N row binding is missing a selected A")
        values = tuple(a_to_n_row[c.a_row] for c in self.selected_candidates)
        _require(all(type(value) is int and value >= 0 for value in values) and len(values) == len(set(values)),
                 "Sparse N rows must be distinct nonnegative actual pool indexes")
        if self.n_rows is not None:
            _require(self.n_rows == values, "Cannot replace an existing sparse N row identity")
        return replace(self, n_rows=values)

    def _payload(self):
        counts = Counter(c.class_id for c in self.selected_candidates)
        by_class = {str(cls): {"retained_rows": count,
                    "retained_groups": len({c.group_id for c in self.selected_candidates if c.class_id == cls}),
                    "retained_polygons": len({c.polygon_id for c in self.selected_candidates if c.class_id == cls}),
                    "class_weight": 1 / len(counts)} for cls, count in sorted(counts.items())}
        return {"schema": SCHEMA, "epoch": self.epoch, "batch_index": self.batch_index,
                "config": self.config.to_dict(), "config_sha256": _digest(self.config.to_dict()),
                "solver": dict(self.solver_metadata), "candidates": [c.to_dict() for c in self.candidates],
                "selected_candidates": [c.to_dict() for c in self.selected_candidates],
                "a_rows": list(self.a_rows), "a_indices": list(self.a_rows),
                "n_rows": None if self.n_rows is None else list(self.n_rows),
                "n_indices": None if self.n_rows is None else list(self.n_rows),
                "n_true_indices": None if self.n_rows is None else list(self.n_rows),
                "n_shuffled_indices": None if self.n_shuffled_indices is None else list(self.n_shuffled_indices),
                "permutation": list(self.permutation), "weights": list(self.weights),
                "weight_numerators": [1] * len(self.weights), "weight_denominators": list(self.weight_denominators),
                "exclusions": list(self.exclusions), "selected_a_slots": list(self.selected_a_slots),
                "candidate_rows": len(self.candidates), "matched_rows": len(self.selected_candidates),
                "by_class": by_class, "matched_no_B": 0,
                "index_domains": {"a_rows": "FULL_NATIVE_CELLS_GT0_POOL_ORDER",
                    "n_rows": "ACTUAL_SPARSE_N_OWN_ROI_CELLS_GT0_POOL_ORDER_OR_NULL_UNTIL_BOUND",
                    "permutation": "COMPACT_SELECTED_S; SHUFFLED_N_EQUALS_TRUE_N[PERMUTATION]"}}

    @property
    def sha256(self):
        return _digest(self._payload())

    def to_dict(self):
        payload = self._payload()
        return {**payload, "sha256": _digest(payload)}


def build_matching(candidates, config=None, *, epoch, batch_index):
    """Maximum-cardinality closed assignment; no features, RNG or N decoding.

    Epoch and batch_index are zero-based.  Returned plans always record and
    hash the exact SciPy version.  A supplied version pin is enforced; when
    absent the observed version becomes the resolved plan's immutable pin.
    """
    import numpy as np
    import scipy
    from scipy.optimize import linear_sum_assignment

    if isinstance(candidates, CandidateCollection):
        candidates = candidates.candidates
    config = config or MatchingConfig()
    if isinstance(config, Mapping):
        config = MatchingConfig(**dict(config))
    _require(isinstance(config, MatchingConfig), "Unknown matching configuration")
    _require(type(epoch) is int and 0 <= epoch < 200 and type(batch_index) is int and 0 <= batch_index < 21,
             "Expected zero-based 200-epoch / 21-batch position")
    _require(config.expected_scipy_version in (None, scipy.__version__), "SciPy version differs from matching pin")
    config = replace(config, expected_scipy_version=scipy.__version__)
    rows = tuple(sorted((c if isinstance(c, CandidateExposure) else CandidateExposure.from_mapping(c)
                         for c in candidates), key=lambda c: (c.a_row, c.source_id, c.polygon_id)))
    _require(len({c.a_row for c in rows}) == len(rows), "Repeated native A pool row")
    _require(len({(c.a_slot, c.polygon_id) for c in rows}) == len(rows), "Repeated Polygon within native A image slot")
    n = len(rows)
    cap = (1 << config.cost_hash_bits) - 1
    discard_cost = n * cap + 1
    # The primary objective dominates the entire possible secondary sum.
    # Float64 represents every finite cost exactly at real native batch sizes.
    _require(n * discard_cost + n * cap < 2**53, "Candidate count exceeds exact integer assignment range")
    cost = np.full((n, n), np.inf, dtype=np.float64)
    for i, left in enumerate(rows):
        cost[i, i] = discard_cost
        for j, right in enumerate(rows):
            if i != j and legal_pair(left, right, config):
                key = {"seed": config.matching_seed, "epoch": epoch, "batch_index": batch_index,
                       "left": left.identity_sha256, "right": right.identity_sha256, "schema": SCHEMA}
                cost[i, j] = int(_digest(key)[:4], 16)
    assigned = dict(zip(*linear_sum_assignment(cost))) if n else {}
    _require(len(assigned) == n and len(set(assigned.values())) == n, "Assignment is not a bijection")
    selected_indexes = tuple(i for i in range(n) if assigned[i] != i)
    selected_set = set(selected_indexes)
    _require({assigned[i] for i in selected_indexes} == selected_set, "Assignment discarded rows without closure")
    compact = {old: new for new, old in enumerate(selected_indexes)}
    permutation = tuple(compact[assigned[i]] for i in selected_indexes)
    selected = tuple(rows[i] for i in selected_indexes)
    _require(all(legal_pair(rows[i], rows[assigned[i]], config) for i in selected_indexes), "Solver used a forbidden edge")
    _require(all(c.has_real_crossgroup_B is not False for c in selected), "Matched row contradicts native B availability")
    classes = Counter(c.class_id for c in selected)
    denominators = tuple(len(classes) * classes[c.class_id] for c in selected)
    weights = tuple(1 / denominator for denominator in denominators)
    _require(all(weights[i] == weights[j] for i, j in enumerate(permutation)), "N weighted marginal changed under permutation")
    bound = [c.n_row is not None for c in selected]
    _require(not any(bound) or all(bound), "Sparse N row binding is incomplete")
    n_rows = tuple(c.n_row for c in selected) if not selected or all(bound) else None
    if n_rows is not None:
        _require(len(n_rows) == len(set(n_rows)), "Two selected A rows refer to one sparse N ROI")
    exclusions = tuple({"a_row": c.a_row, "polygon_id": c.polygon_id,
                        "reason": "NO_LEGAL_EDGE" if not np.isfinite(np.delete(cost[i], i)).any()
                            else "MAXIMUM_CLOSED_SUBSET_DISCARD"}
                       for i, c in enumerate(rows) if i not in selected_set)
    solver = {"name": "scipy.optimize.linear_sum_assignment", "scipy_version": scipy.__version__,
              "numpy_version": np.__version__, "row_order": "A_ROW_SOURCE_POLYGON_ASCENDING",
              "tie_policy": "PINNED_SOLVER_WITH_STABLE_CANONICAL_ROWS_AND_16_BIT_SHA_COSTS",
              "unique_optimum_claimed": False, "secondary_cost_max": cap, "diagonal_discard_cost": discard_cost,
              "objective": "MAX_NON_SELF_ROWS_THEN_MIN_SHA_SECONDARY_COST", "unmatched_edges": "DIAGONAL_ONLY"}
    return MatchingPlan(rows, selected, permutation, weights, denominators, n_rows, exclusions,
                        config, epoch, batch_index, solver)
