"""CPU checks for scope, all-GT preservation and exact-pixel historical evidence."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.exp17e.census import Files, extract_polygons, review_evidence, sha, validate_scope


class CensusTests(unittest.TestCase):
    def test_all_original_polygons_retained_beyond_selected_targets(self):
        annotation={"imageWidth":20,"imageHeight":10,"shapes":[{"shape_type":"polygon","label":name,"points":[[1,1],[3,1],[2,3]]}
                     for name in ("Burn","Crack")]}
        source={"selected_polygon_ids":'["x_000"]',"all_gt_polygon_ids":'["x_000","x_001"]'}
        rows=extract_polygons("x","near_1",annotation,[20,10],source)
        self.assertEqual(len(rows),2);self.assertTrue(rows[1]["extra_original_annotation_not_selected_by_a9"])
        self.assertEqual(rows[1]["id_basis"],"DERIVED_ID")
        self.assertEqual(rows[1]["class_id"],1)

    def test_invalid_coordinates_and_unknown_classes_are_not_silently_dropped(self):
        source={"selected_polygon_ids":'["x_000"]',"all_gt_polygon_ids":'["x_000"]'}
        for label,points in (("unknown",[[1,1],[2,1],[1,2]]),("Burn",[[1,1],[21,1],[1,2]])):
            with self.assertRaises(ValueError):
                extract_polygons("x","g",{"imageWidth":20,"imageHeight":10,"shapes":[{"shape_type":"polygon","label":label,"points":points}]},[20,10],source)

    def test_manifest_scope_rejects_duplicates_and_excluded_source_overlap(self):
        with self.assertRaisesRegex(ValueError,"379 unique"):
            validate_scope([{"source_id":"1"}]*379,[])
        types=["ALIGNED"]*346+["RAW"]*29+["BOUNDED"]+["CROP"]*3
        rows=[{"source_id":str(i),"group_id":str(i%359),"asset_type":kind,"split":"train"} for i,kind in enumerate(types)]
        excluded=[{"source_id":"excluded_"+str(i)} for i in range(13)]
        validate_scope(rows,excluded)
        excluded[0]={"source_id":"0"}
        with self.assertRaisesRegex(ValueError,"Excluded 13"):
            validate_scope(rows,excluded)

    def test_raw_same_version_does_not_inherit_aligned_pixel_acceptance(self):
        raw={"polygon_decisions":'{"p":"PASS"}',"decision":"PASS","b_region_review":"CONFIRMED","whole_image_normal":"YES","notes":"","reason":""}
        variant={"resolved_polygon_decisions":'{"p":"PASS"}',"review_id":"rid","variant_id":"v","version":"V1","aligned_sha256":"aligned","ai_decision":"PASS"}
        aligned=review_evidence(raw,variant,"p",{},"ALIGNED","aligned",{})
        unaligned=review_evidence(raw,variant,"p",{},"RAW","raw",{})
        self.assertTrue(aligned["exact_current_n_region_acceptance_evidence"])
        self.assertTrue(unaligned["same_review_user_polygon_pass_and_b_confirmed"])
        self.assertFalse(unaligned["exact_current_n_region_acceptance_evidence"])

    def test_unreviewed_extra_polygon_not_promoted_by_whole_image_pass(self):
        raw={"polygon_decisions":"{}","decision":"PASS","b_region_review":"CONFIRMED","whole_image_normal":"YES","notes":"","reason":""}
        variant={"resolved_polygon_decisions":'{"p":"PASS"}',"review_id":"rid","variant_id":"v","version":"V1","aligned_sha256":"aligned","ai_decision":"PASS"}
        evidence=review_evidence(raw,variant,"extra",{},"ALIGNED","aligned",{})
        self.assertFalse(evidence["same_review_user_polygon_pass_and_b_confirmed"])
        self.assertEqual(evidence["polygon_state_basis"],"NOT_IN_REVIEW_TARGET_SET")

    def test_current_hash_failure_stops_and_identical_content_has_correct_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"asset.bin";p.write_bytes(b"original")
            files=Files();item=files.check(p,sha(p),"selected_N")
            self.assertEqual(item["bytes"],8)
            p.write_bytes(b"modified")
            with self.assertRaises(ValueError):files.postcheck()
            with self.assertRaisesRegex(ValueError,"SHA256 differs"):
                Files().check(p,"0"*64,"selected_N")


if __name__=="__main__":unittest.main()
