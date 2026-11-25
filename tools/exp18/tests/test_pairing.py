"""Small exhaustive matching oracles and population/weight invariants."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

import scipy

from tools.exp18.pairing import (
    CandidateExposure, MatchingConfig, build_matching, collect_candidates, legal_pair,
)


def candidate(row, group=None, *, cls=0, family="V1", area=.01, cells=8):
    return CandidateExposure(a_row=row, a_slot=row, source_id=f"s{row}", group_id=group or f"g{row}",
        polygon_id=f"p{row}", class_id=cls, repair_version=family, n_sha256=f"{row + 1:064x}",
        source_area_fraction=area, p3_cells=cells, augmentation_sha256="a" * 64, mask_sha256="b" * 64)


def brute_cardinality(rows):
    best = 0
    for permutation in itertools.permutations(range(len(rows))):
        if all(i == j or legal_pair(rows[i], rows[j]) for i, j in enumerate(permutation)):
            best = max(best, sum(i != j for i, j in enumerate(permutation)))
    return best


class PairingTests(unittest.TestCase):
    def plan(self, rows, **kwargs):
        return build_matching(rows, epoch=kwargs.pop("epoch", 0), batch_index=kwargs.pop("batch_index", 0), **kwargs)

    def assert_closed(self, plan):
        n = len(plan.a_rows)
        self.assertEqual(set(plan.permutation), set(range(n)))
        self.assertTrue(all(i != j for i, j in enumerate(plan.permutation)))
        for i, j in enumerate(plan.permutation):
            self.assertTrue(legal_pair(plan.selected_candidates[i], plan.selected_candidates[j]))

    def test_empty_and_one_group_are_valid_empty_plans(self):
        for rows in ([], [candidate(0)], [candidate(i, "same") for i in range(4)]):
            plan = self.plan(rows)
            self.assertEqual(plan.a_rows, ())
            self.assertEqual(plan.n_rows, ())
            self.assertEqual(plan.weights, ())
            self.assert_closed(plan)

    def test_odd_cycle_keeps_three_rows(self):
        plan = self.plan([candidate(i) for i in range(3)])
        self.assertEqual(len(plan.a_rows), 3)
        self.assert_closed(plan)

    def test_dominant_group_retains_closed_maximum_subset(self):
        rows = [candidate(i, "major" if i < 4 else f"minor{i}") for i in range(6)]
        plan = self.plan(rows)
        self.assertEqual(len(plan.a_rows), 4)
        self.assertEqual(Counter(c.group_id for c in plan.selected_candidates)["major"], 2)
        self.assert_closed(plan)

    def test_bruteforce_oracle_for_small_geometry_restricted_cases(self):
        rng = random.Random(18)
        for n in range(2, 8):
            for trial in range(4):
                rows = [candidate(i, f"g{rng.randrange(3)}", cls=rng.randrange(2),
                                  family=f"V{rng.randrange(1, 3)}", area=rng.choice([.01, .02, .045]),
                                  cells=rng.choice([1, 2, 4, 5, 8, 18])) for i in range(n)]
                with self.subTest(n=n, trial=trial):
                    plan = self.plan(rows)
                    self.assertEqual(len(plan.a_rows), brute_cardinality(rows))
                    self.assert_closed(plan)

    def test_ratio_boundaries_are_symmetric_and_inclusive(self):
        left = candidate(0, cells=5)
        right = candidate(1, area=.02, cells=10)
        self.assertTrue(legal_pair(left, right))
        self.assertTrue(legal_pair(right, left))
        self.assertFalse(legal_pair(left, replace(right, source_area_fraction=.02000001)))
        self.assertFalse(legal_pair(left, replace(right, p3_cells=11)))
        self.assertFalse(legal_pair(candidate(0, cells=4), candidate(1, cells=5)))
        self.assertFalse(legal_pair(candidate(0, cells=1), candidate(1, cells=2)))

    def test_class_family_group_are_hard_constraints(self):
        base = candidate(0)
        for other in (candidate(1, cls=1), candidate(1, family="V2"), candidate(1, group=base.group_id)):
            self.assertFalse(legal_pair(base, other))
            self.assertEqual(self.plan([base, other]).a_rows, ())

    def test_class_equal_weights_preserve_both_marginals_with_unequal_group_counts(self):
        rows = [candidate(0, "a"), candidate(1, "a"), candidate(2, "b"), candidate(3, "c"),
                candidate(4, "d", cls=1), candidate(5, "e", cls=1)]
        plan = self.plan(rows)
        self.assertEqual(len(plan.a_rows), 6)
        self.assert_closed(plan)
        a_mass, true_n_mass, shuffled_n_mass, class_mass = {}, {}, {}, Counter()
        for i, (row, denominator) in enumerate(zip(plan.selected_candidates, plan.weight_denominators)):
            weight = Fraction(1, denominator)
            a_mass[row.a_row] = weight
            true_n_mass[row.a_row] = weight
            shuffled_n_mass[plan.selected_candidates[plan.permutation[i]].a_row] = weight
            class_mass[row.class_id] += weight
        self.assertEqual(a_mass, true_n_mass)
        self.assertEqual(true_n_mass, shuffled_n_mass)
        self.assertEqual(set(class_mass.values()), {Fraction(1, 2)})

    def test_explicit_sparse_n_index_binding_never_assumes_a_row_identity(self):
        plan = self.plan([candidate(7), candidate(19)])
        self.assertIsNone(plan.n_rows)
        bound = plan.bind_n_rows({7: 2, 19: 0, 100: 99})
        self.assertEqual(bound.a_indices, (7, 19))
        self.assertEqual(bound.n_indices, (2, 0))
        self.assertEqual(bound.n_shuffled_indices, (0, 2))
        self.assertNotEqual(plan.sha256, bound.sha256)
        self.assertEqual(bound.bind_n_rows({7: 2, 19: 0}).sha256, bound.sha256)
        with self.assertRaises(ValueError):
            bound.bind_n_rows({7: 0, 19: 2})
        with self.assertRaises(ValueError):
            plan.bind_n_rows({7: 0, 19: 0})

    def test_canonical_input_and_crossprocess_determinism(self):
        rows = [candidate(i, f"g{i % 3}") for i in range(6)]
        plan = self.plan(rows, epoch=199, batch_index=20)
        self.assertEqual(plan.sha256, self.plan(list(reversed(rows)), epoch=199, batch_index=20).sha256)
        payload = json.dumps([c.to_dict() for c in rows])
        script = ("import json,sys; from tools.exp18.pairing import build_matching; "
                  "print(build_matching(json.loads(sys.stdin.read()),epoch=199,batch_index=20).sha256)")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="123")
        run = subprocess.run([sys.executable, "-B", "-c", script], input=payload, text=True,
                             cwd=str(Path(__file__).resolve().parents[3]), env=environment,
                             capture_output=True, check=True)
        self.assertEqual(run.stdout.strip(), plan.sha256)

    def test_seed_scope_and_version_are_recorded_and_bound(self):
        rows = [candidate(i) for i in range(5)]
        plan = self.plan(rows)
        self.assertEqual(plan.config.expected_scipy_version, scipy.__version__)
        self.assertEqual(plan.solver_metadata["scipy_version"], scipy.__version__)
        self.assertFalse(plan.solver_metadata["unique_optimum_claimed"])
        self.assertNotEqual(plan.sha256, self.plan(rows, epoch=1).sha256)
        self.assertNotEqual(plan.sha256, self.plan(rows, config=MatchingConfig(matching_seed=1)).sha256)
        with self.assertRaises(ValueError):
            self.plan(rows, config=MatchingConfig(expected_scipy_version="impossible-version"))

    def test_protocol_mapping_validates_flags_and_uses_explicit_namespace_seed(self):
        from tools.exp18.protocol import make_protocol
        protocol = make_protocol()
        config = MatchingConfig.from_protocol(protocol)
        self.assertEqual(config.matching_seed, 42)
        protocol["matching"]["unknown_silent_override"] = True
        with self.assertRaises(ValueError):
            MatchingConfig.from_protocol(protocol)
        protocol = make_protocol()
        protocol["matching"]["closed_permutation"] = False
        with self.assertRaises(ValueError):
            MatchingConfig.from_protocol(protocol)

    def test_duplicate_or_invalid_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            self.plan([candidate(0), candidate(0)])
        for change in ({"p3_cells": 0}, {"source_area_fraction": float("nan")}, {"repair_version": "V_UNKNOWN"},
                       {"augmentation_sha256": ""}, {"class_id": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(candidate(0), **change)

    def test_collect_no_B_is_not_removed_before_matching(self):
        first = candidate(0)
        row = {"row_index": 0, "slot": 0, "source_id": first.source_id, "group_id": first.group_id,
               "polygon_id": first.polygon_id, "class_id": 0, "p3_cells": 8, "augmentation_sha256": "a" * 64}
        trusted = {first.polygon_id: {**row, "repair_version": "V1", "n_sha256": "c" * 64}}
        support = {0: {"polygon_id": first.polygon_id, "support_complete": True, "mask_sha256": "b" * 64}}
        result = collect_candidates([row], trusted, support, {first.polygon_id: {"source_area_fraction": .01}})
        self.assertEqual(len(result.candidates), 1)
        self.assertFalse(result.candidates[0].has_real_crossgroup_B)
        self.assertEqual(result.counts["candidate_no_B"], 1)
        self.assertEqual(self.plan(result).a_rows, ())

    def test_collect_support_unknown_mismatched_and_incomplete(self):
        row = {"row_index": 0, "slot": 0, "source_id": "s0", "group_id": "g0", "polygon_id": "p0",
               "class_id": 0, "p3_cells": 8, "augmentation_sha256": "a" * 64}
        trusted = {"p0": {**row, "repair_version": "V1", "n_sha256": "c" * 64}}
        area = {"p0": .01}
        with self.assertRaises(ValueError):
            collect_candidates([row], trusted, {}, area)
        with self.assertRaises(ValueError):
            collect_candidates([row], trusted, {0: {"polygon_id": "wrong", "support_complete": True}}, area)
        incomplete = collect_candidates([row], trusted,
                                        {0: {"polygon_id": "p0", "support_complete": False}}, area)
        self.assertEqual(incomplete.candidates, ())
        self.assertEqual(incomplete.exclusions[0]["reason"], "INCOMPLETE_A_N_VALID_SUPPORT")


if __name__ == "__main__":
    unittest.main()
