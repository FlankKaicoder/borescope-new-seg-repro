import unittest

from tools.exp17e.coverage import compute, concentration


def polygon(source='a', group='group-a', sha='1' * 64):
    return {'source_id': source, 'group_id': group, 'class_name': 'Crack', 'class_id': 1,
            'n_asset_sha256': sha, 'repair_version': 'V1', 'asset_type': 'ALIGNED',
            'mapping_verified': True, 'support_complete': True, 'primary_status': 'CURRENT_INHERITED'}


def train(source, group):
    return {'stem': source, 'group_id': group, 'split': 'train', 'labels_present': 'Crack'}


class CoverageTests(unittest.TestCase):
    def test_b_outside_a10_scope_remains_available(self):
        report = compute([polygon()], [train('a', 'group-a'), train('outside379', 'group-b')])
        self.assertEqual(report['classes']['Crack']['potential_a_b_n_ranking_pending_review_and_data_gate']['polygons'], 1)

    def test_near_duplicate_images_are_not_independent_b(self):
        report = compute([polygon()], [train('a', 'group-a'), train('another', 'group-a')])
        self.assertEqual(report['classes']['Crack']['full_train_b_independent_groups'], 1)
        self.assertEqual(report['classes']['Crack']['potential_a_b_n_ranking_pending_review_and_data_gate']['polygons'], 0)

    def test_multiple_polygons_do_not_inflate_independent_groups(self):
        c = concentration([polygon(), polygon(), polygon('b', 'group-b')])
        self.assertEqual(c['independent_groups'], 2)
        self.assertAlmostEqual(c['group_share_hhi'], 5 / 9)

    def test_same_source_cannot_mix_n_versions(self):
        with self.assertRaisesRegex(ValueError, 'more than one A10'):
            compute([polygon(), polygon(sha='2' * 64)], [train('a', 'group-a')])

    def test_failed_state_cannot_enter_ranking_via_inconsistent_boolean(self):
        p = polygon(); p['primary_status'] = 'TECHNICAL_SUPPORT_FAILURE'
        r = compute([p], [train('a', 'group-a'), train('outside379', 'group-b')])
        self.assertEqual(r['classes']['Crack']['potential_a_b_n_ranking_pending_review_and_data_gate']['polygons'], 0)

    def test_frozen_group_identity_must_match(self):
        with self.assertRaisesRegex(ValueError, 'frozen TRAIN identity'):
            compute([polygon(group='wrong-group')], [train('a', 'group-a')])


if __name__ == '__main__':
    unittest.main()
