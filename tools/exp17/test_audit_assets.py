"""CPU tests of integrity and independent-source counting gates."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from tools.exp17.audit_assets import Files, numeric_pass, region_metrics
from tools.exp17.audit_coverage import coverage


class TestAssetAudit(unittest.TestCase):
    def test_roi_integer_support_and_context(self):
        target=Image.new('L',(40,40),0)
        ImageDraw.Draw(target).rectangle((18,18,21,21),fill=255)
        valid=Image.new('L',target.size,255)
        metric=region_metrics(target,valid)
        self.assertEqual(metric['target_pixels'],16)
        self.assertEqual(metric['target_bbox'],[18,18,22,22])
        self.assertEqual(metric['roi'],[6,6,34,34])
        self.assertEqual(metric['context_pixels'],28*28-16)
        self.assertTrue(numeric_pass(metric))
        valid.putpixel((18,18),0)
        self.assertEqual(region_metrics(target,valid)['unsupported_target_pixels'],1)
        self.assertFalse(numeric_pass(region_metrics(target,valid)))

    def test_screen_boundary_and_independent_requirements(self):
        m={'unsupported_target_pixels':0,'valid_roi_pixels':95,'roi_pixels':100,
           'valid_context_pixels':90,'context_pixels':100}
        self.assertTrue(numeric_pass(m))
        self.assertFalse(numeric_pass({**m,'valid_roi_pixels':94}))
        self.assertFalse(numeric_pass({**m,'valid_context_pixels':89}))
        self.assertFalse(numeric_pass({**m,'unsupported_target_pixels':1}))

    def test_changed_file_is_detected_after_read(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'artifact';p.write_bytes(b'original')
            files=Files();files.check(p);files.postcheck()
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):files.postcheck()

    def test_near_duplicates_and_variants_do_not_inflate_groups(self):
        def row(source,group):
            return {'g0_status':'G0_PASS','source_id':source,'group_id':group,'class':'Dent',
                    'variants':[{},{}],'risk_flags':['PENDING'],'g1_status':'G1_VISUAL_PENDING','numeric_screen_pass':True}
        source,group,summary=coverage([row('a','near1'),row('b','near1'),row('c','singleton')])
        self.assertEqual(len(source),3);self.assertEqual(len(group),2)
        self.assertEqual(summary['independent_source_groups'],2)
        self.assertEqual(summary['class_source_groups_overlapping']['Dent'],2)
        self.assertEqual(summary['formal_trusted_pairs'],0)
        self.assertEqual(sum(x['variant_polygon_combinations'] for x in source),6)


if __name__=='__main__':unittest.main()
