import unittest

from tools.exp17e.expansion import classify, current_d_question, inheritance_route, verify_polygon_mapping


def state(**changes):
    flags = dict(identity_ok=True, asset_exists=True, mapping_verified=True, support_complete=True,
                 historical_rejected=False, region_evidence=True, current_specific_question=False,
                 proposal_ineligible=False, had_b545=True)
    return classify(**(flags | changes))[0]


class ExpansionTests(unittest.TestCase):
    def test_unchanged_accepted_region_does_not_require_repeated_review(self):
        self.assertEqual(state(), 'CURRENT_INHERITED')

    def test_a10_whole_image_selection_cannot_replace_region_evidence(self):
        self.assertEqual(state(region_evidence=False), 'NEW_HUMAN_REVIEW_CANDIDATE')

    def test_geometry_failure_cannot_be_overridden_by_history(self):
        self.assertEqual(state(mapping_verified=False), 'MAPPING_UNVERIFIED')
        self.assertEqual(state(support_complete=False), 'TECHNICAL_SUPPORT_FAILURE')

    def test_user_rejection_requires_new_explicit_decision(self):
        self.assertEqual(state(historical_rejected=True), 'HISTORICAL_USER_REJECTED')

    def test_old_ai_conflict_is_not_a_user_rejection_or_automatic_exclusion(self):
        self.assertEqual(state(proposal_ineligible=True, had_b545=False), 'NEW_HUMAN_REVIEW_CANDIDATE')
        self.assertEqual(state(proposal_ineligible=True), 'CURRENT_TARGETED_REVIEW')

    def test_old_question_on_another_version_does_not_transfer(self):
        p = {'exp17d_record': {'data_gate_disposition': 'TARGETED_EVIDENCE_REVIEW', 'review_id': 'old'}, 'n_review_id': 'new'}
        self.assertFalse(current_d_question(p, 'EXACT_REVIEWED_ALIGNED_REGION'))

    def test_same_size_raw_is_not_identity_or_pixel_acceptance(self):
        p = {'same_n_version_history': {'same_review_user_polygon_pass_and_b_confirmed': True,
                                      'historical_aligned_sha256': 'old-aligned'},
             'n_asset_type': 'RAW', 'n_asset_sha256': 'current-raw'}
        sm = {'mapping_verified': True, 'historical_raw_to_source': [1],
              'evidence': {'historical_raw_sha256': 'another-raw', 'historical_aligned_sha256': 'old-aligned'}}
        self.assertIsNone(inheritance_route(p, {}, sm, {'support_complete': True}, {}))
        sm['evidence']['historical_raw_sha256'] = 'current-raw'
        self.assertEqual(inheritance_route(p, {}, sm, {'support_complete': True}, {}), 'RAW_INFERENTIAL_REGION_INHERITANCE_VIA_FROZEN_H')

    def test_polygon_geometry_cannot_borrow_another_matrix_or_class(self):
        p = dict(source_id='a', group_id='g', polygon_id='a_000', class_id=1, class_name='Crack', n_asset_sha256='n')
        sm = {'matrix_identity_sha256': 'correct'}
        pm = p | {'matrix_identity_sha256': 'correct'}
        verify_polygon_mapping(p, sm, pm)
        with self.assertRaisesRegex(ValueError, 'matrix identity'):
            verify_polygon_mapping(p, sm, pm | {'matrix_identity_sha256': 'another'})
        with self.assertRaisesRegex(ValueError, 'matrix identity'):
            verify_polygon_mapping(p, sm, pm | {'class_id': 0})


if __name__ == '__main__':
    unittest.main()
