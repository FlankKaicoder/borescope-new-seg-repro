"""Synthetic-only adversarial review binding tests; no actual candidate is approved."""
import copy
import unittest

from tools.exp17.review_package import NEGATIVE, POSITIVE, new_review, normalize, validate_export, validate_review


def fixture():
    return normalize({"candidate_id":"SYNTHETIC_ONLY","source_id":"synthetic","group_id":"synthetic",
      "polygon_id":"synthetic_000","class":"Crack","review_id":"synthetic_review","variant_id":"synthetic_V1",
      "version":"V1","original_sha256":"1"*64,"annotation_sha256":"2"*64,"aligned_sha256":"3"*64,
      "valid_mask_sha256":"4"*64,"geometry_sha256":"5"*64,"matrix_output_to_source":[[1,0,0],[0,1,0]],
      "polygon_coordinates":[[1,1],[2,1],[2,2]],"roi":[0,0,3,3],"g0_status":"G0_PASS"})


class ReviewBinding(unittest.TestCase):
    def test_no_historical_pass_inheritance(self):
        row=fixture();row["history"]={"human":{"final_decision":"PASS"}}
        result=new_review(row,"package")
        self.assertEqual(result["final_decision"],"PENDING")
        self.assertEqual(result["review_role"],"UNREVIEWED")
        self.assertFalse(result["training_approved"])
        self.assertTrue(validate_review(row,result,"package"))

    def test_asset_and_version_tampering(self):
        row=fixture();v=new_review(row,"package")
        for key in ["aligned_sha256","valid_mask_sha256","version"]:
            changed=copy.deepcopy(v);changed["asset_identity"][key]="changed"
            with self.assertRaises(ValueError):validate_review(row,changed,"package")
        for key in ["source_id","group_id","polygon_id","class","review_id","variant_id","version","schema"]:
            changed=copy.deepcopy(v);changed[key]="changed"
            with self.assertRaises(ValueError):validate_review(row,changed,"package")

    def test_pass_requires_complete_attributed_independent_review(self):
        row=fixture();v=new_review(row,"package");v["final_decision"]="PASS"
        with self.assertRaises(ValueError):validate_review(row,v,"package")
        v.update({k:"YES" for k in POSITIVE});v.update({k:"NO" for k in NEGATIVE})
        v.update(reviewer_id="SYNTHETIC_QA_NOT_HUMAN",boundary_evidence_note="synthetic independent edge",
                 reviewed_at="2026-11-22T00:00:00Z",review_role="USER_LOCAL_REVIEW_UNVERIFIED")
        self.assertTrue(validate_review(row,v,"package"))
        v["other_defect_contamination"]="UNCERTAIN"
        with self.assertRaises(ValueError):validate_review(row,v,"package")

    def test_no_training_gate_or_duplicate_export(self):
        row=fixture();v=new_review(row,"package");bundle={"package_id":"package","records":[row]}
        payload={"schema":v["schema"],"package_id":"package","training_approved":False,
                 "independent_data_gate":"NOT_GRANTED","records":[v]}
        self.assertEqual(validate_export(bundle,payload)["formal_trusted_pairs"],0)
        payload["records"].append(copy.deepcopy(v))
        with self.assertRaises(ValueError):validate_export(bundle,payload)
        payload["records"]=[v];payload["training_approved"]=True
        with self.assertRaises(ValueError):validate_export(bundle,payload)


if __name__=="__main__":unittest.main()
