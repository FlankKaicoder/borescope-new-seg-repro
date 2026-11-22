"""CPU checks of immutable decision merge, precise scope and evidence tiers."""
import copy
import tempfile
import unittest
from pathlib import Path

from tools.exp17f.reconcile import (ACCEPT,REJECT,HISTORICAL,USER_ACCEPTED,USER_REJECTED,TECHNICAL,OUTSIDE,
    Inputs,counts,evidence_tier,final_state,join_decisions,raw_direction_valid,sha)


def original(pid, primary="NEW_HUMAN_REVIEW_CANDIDATE", kind="ALIGNED", route="EXACT_REVIEWED_ALIGNED_REGION"):
    return {"candidate_id":"candidate_"+pid,"source_id":"185","group_id":"near_group","polygon_id":pid,
            "class_id":6,"class_name":"corrosion","primary_status":primary,"asset_type":kind,"evidence_route":route,
            "arbitrary_original_history":{"human":"previous value","notes":["must remain",1.]}}


def reviews_for(rows,decisions):
    records=[{"candidate_id":r["candidate_id"],"decision":d,"user_notes":"中文\n多行"} for r,d in zip(rows,decisions)]
    return records,copy.deepcopy(records),[{"candidate_id":r["candidate_id"],"physical_line_start":i+2} for i,r in enumerate(records)]


class ReconcileTests(unittest.TestCase):
    def test_inherited_is_not_a_new_user_acceptance_and_original_is_unchanged(self):
        old=original("p","CURRENT_INHERITED");before=copy.deepcopy(old)
        row=join_decisions([old],[],[],[])[0]
        self.assertEqual(row["original"],before)
        self.assertEqual(old,before)
        self.assertEqual(row["final_state"],HISTORICAL)
        self.assertIsNone(row["new_user_review"])
        self.assertFalse(row["training_approved"])

    def test_source185_reject_does_not_spread_to_two_accepted_siblings(self):
        originals=[original(p) for p in ("185_005","185_002","185_003")]
        records,raw,audit=reviews_for(originals,[REJECT,ACCEPT,ACCEPT])
        merged=join_decisions(originals,records,raw,audit)
        self.assertEqual([m["final_state"] for m in merged],[USER_REJECTED,USER_ACCEPTED,USER_ACCEPTED])
        self.assertEqual(sum(m["candidate_pool_pending_data_gate"] for m in merged),2)
        self.assertEqual(merged[0]["new_user_review"]["raw_typed_unchanged"]["user_notes"],"中文\n多行")

    def test_technical_lock_cannot_be_accepted(self):
        old=original("p","TECHNICAL_SUPPORT_FAILURE")
        self.assertEqual(final_state(old,None),TECHNICAL)
        with self.assertRaisesRegex(ValueError,"technically locked"):
            final_state(old,{"decision":ACCEPT})

    def test_review_cannot_overwrite_existing_inheritance(self):
        with self.assertRaisesRegex(ValueError,"outside the frozen"):
            final_state(original("p","CURRENT_INHERITED"),{"decision":ACCEPT})

    def test_unreviewed_targeted_does_not_auto_enter_pool(self):
        self.assertEqual(final_state(original("p","CURRENT_TARGETED_REVIEW"),None),OUTSIDE)

    def test_historical_raw_and_new_raw_user_acceptance_are_distinct_tiers(self):
        old=original("p","CURRENT_INHERITED","RAW","RAW_INFERENTIAL_REGION_INHERITANCE_VIA_FROZEN_H")
        self.assertEqual(evidence_tier(old,HISTORICAL),"RAW_INFERENTIAL_REGION_INHERITANCE_VIA_FROZEN_H")
        self.assertEqual(evidence_tier(old,USER_ACCEPTED),"NEW_USER_ACCEPTED_CURRENT_RAW_POLYGON")

    def test_raw_mapping_checks_historical_direction_and_inverse(self):
        h=[[2.,0.,4.],[0.,3.,6.],[0.,0.,1.]]
        inv=[[.5,0.,-2.],[0.,1/3,-2.],[0.,0.,1.]]
        self.assertTrue(raw_direction_valid({"historical_raw_to_source":h,"n_to_source":h,"source_to_n":inv}))
        self.assertFalse(raw_direction_valid({"historical_raw_to_source":h,"n_to_source":inv,"source_to_n":h}))

    def test_independent_group_counts_do_not_inflate_with_polygons_or_sources(self):
        rows=[{"source_id":"a","group_id":"near1"},{"source_id":"a","group_id":"near1"},
              {"source_id":"b","group_id":"near1"},{"source_id":"c","group_id":"unique"}]
        self.assertEqual(counts(rows),{"polygons":4,"source_images":3,"independent_source_groups":2})

    def test_metadata_identity_changes_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"input.json";p.write_bytes(b"original")
            inputs=Inputs();inputs.check(p,sha(p));p.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError,"changed during"):
                inputs.postcheck()


if __name__=="__main__":unittest.main()
