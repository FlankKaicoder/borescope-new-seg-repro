"""Meaningful CPU tests for historical inheritance and non-propagation."""
import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.exp17d.reconcile import (ASSET_ONLY, EXCLUDED, INHERITED, TARGETED, UNKNOWN,
                                    count, disposition, historical_state, read_csv, resolve_polygon)
from tools.exp17d.reconcile import main, require_semantic_rules


def human(overall="PASS", explicit=None, b="CONFIRMED", whole="YES"):
    return {"decision":overall,"polygon_decisions":json.dumps(explicit or {}),"b_region_review":b,
            "whole_image_normal":whole,"final_version_choice":""}


class ReconciliationTests(unittest.TestCase):
    def test_explicit_local_rejection_overrides_whole_default(self):
        row=human(explicit={"p":"RETRY_REPAIR"})
        state,basis=resolve_polygon(row,{"p":"RETRY_REPAIR"},"p")
        self.assertEqual((state,basis),("RETRY_REPAIR","EXPLICIT"))
        self.assertEqual(historical_state(row,state),"HISTORICAL_USER_REJECTED")

    def test_explicit_final_exclusion_precedes_semantic_acceptance(self):
        row=human();row["final_version_choice"]="EXCLUDE"
        self.assertEqual(historical_state(row,"PASS"),"HISTORICAL_USER_REJECTED")
        self.assertEqual(disposition(historical_state(row,"PASS"),False,[],[]),EXCLUDED)

    def test_default_is_only_for_frozen_same_review_targets_and_confirmed_region(self):
        row=human()
        self.assertEqual(resolve_polygon(row,{"p":"PASS"},"p"),("PASS","OVERALL_DEFAULT"))
        with self.assertRaisesRegex(ValueError,"frozen target"):
            resolve_polygon(row,{"p":"PASS"},"another_polygon")
        self.assertEqual(historical_state(human(b=""),"PASS"),"HISTORICAL_USER_PENDING")

    def test_local_pass_survives_overall_alignment_retry_with_exact_scope(self):
        row=human(overall="RETRY_ALIGNMENT",explicit={"p":"PASS"},whole="")
        state,_=resolve_polygon(row,{"p":"PASS","q":"RETRY_ALIGNMENT"},"p")
        self.assertEqual(historical_state(row,state),"HISTORICAL_USER_ACCEPTED_WITH_SCOPE")

    def test_asset_pending_is_not_rejection_and_blank_final_choice_not_gate(self):
        state=historical_state(human(),"PASS")
        self.assertEqual(disposition(state,True,[],[]),ASSET_ONLY)
        self.assertEqual(disposition(state,False,[],[]),INHERITED)

    def test_targeted_specific_evidence_keeps_acceptance_but_blocks_inheritance(self):
        state=historical_state(human(),"PASS")
        self.assertEqual(state,"HISTORICAL_USER_ACCEPTED")
        self.assertEqual(disposition(state,False,[{"type":"TARGETED_REVIEW"}],[]),TARGETED)
        self.assertEqual(disposition(state,False,[],["wrong_same_version_hash"]),UNKNOWN)

    def test_other_version_rejection_does_not_propagate(self):
        rejected=historical_state(human(overall="RETRY_REPAIR",whole=""),"RETRY_REPAIR")
        accepted=historical_state(human(),"PASS")
        self.assertEqual(disposition(rejected,False,[],[]),EXCLUDED)
        self.assertEqual(disposition(accepted,False,[],[]),INHERITED)

    def test_multiline_notes_physical_and_record_rows_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"human.csv"
            with p.open("w",encoding="utf-8",newline="") as stream:
                writer=csv.writer(stream);writer.writerow(["review_id","notes"])
                writer.writerow(["a","line1\nline2"]);writer.writerow(["b","one line"])
            rows=read_csv(p)
            self.assertEqual((rows[0]["physical_line_start"],rows[0]["physical_line_end"]),(2,3))
            self.assertEqual((rows[1]["data_row_1based"],rows[1]["record_row_including_header_1based"],rows[1]["physical_line_start"]),(2,3,4))

    def test_source_group_counts_do_not_inflate_with_polygons_or_variants(self):
        rows=[{"source_id":"a","group_id":"near_1"},{"source_id":"a","group_id":"near_1"},
              {"source_id":"b","group_id":"near_1"},{"source_id":"c","group_id":"singleton_c"}]
        self.assertEqual(count(rows),{"polygons":4,"source_ids":3,"independent_source_groups":2})

    def test_missing_semantic_evidence_cli_fails_before_output_creation(self):
        with patch("sys.argv",["reconcile","--a9-root","unused","--exp17-audit","unused","--output","unused"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code,2)

    def test_empty_or_incomplete_semantic_rulebook_is_not_formal_complete(self):
        for value in ({},{"rules":[]},{"rules":[{"type":"TARGETED_REVIEW"}]}):
            with self.assertRaises(ValueError):
                require_semantic_rules(value)


if __name__=="__main__":unittest.main()
