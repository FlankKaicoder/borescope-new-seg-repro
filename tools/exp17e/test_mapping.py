"""Synthetic CPU contracts for historical transform reuse and support checks."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.exp17e.mapping import IDENTITY, inverse, polygon_mapping, raster, transform_points, warp_valid_nearest


class MappingTests(unittest.TestCase):
    def test_inverse_is_source_to_raw_not_raw_to_source(self):
        raw_to_source=[[.5,0.,10.],[0.,.25,-3.]]
        source=[[12.,1.],[30.,12.]]
        raw=transform_points(source,inverse(raw_to_source))
        self.assertEqual(raw,[[4.,16.],[40.,60.]])
        self.assertEqual(transform_points(raw,raw_to_source),source)
        self.assertNotEqual(raw,transform_points(source,raw_to_source))

    def test_nearest_mask_pixel_centers_and_translation(self):
        m=Image.new('L',(8,6),0);m.putpixel((2,3),255)
        same=warp_valid_nearest(m,m.size,IDENTITY)
        self.assertEqual(same.tobytes(),m.tobytes())
        shifted=warp_valid_nearest(m,m.size,[[1.,0.,-1.],[0.,1.,0.]])
        self.assertEqual(shifted.getpixel((3,3)),255)
        self.assertEqual(shifted.histogram()[255],1)
        scaled=warp_valid_nearest(m,(16,12),[[.5,0.,0.],[0.,.5,0.]])
        self.assertEqual(set(scaled.tobytes()),{0,255})
        self.assertEqual(scaled.histogram()[255],4)

    def test_crop_translation_support_and_outside_target(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'valid.png';Image.new('L',(6,5),255).save(path)
            mapping={'n_asset_sha256':'a'*64,'mapping_verified':True,'matrix_identity_sha256':'b'*64,
                     'source_to_n':[[1.,0.,-2.],[0.,1.,-1.],[0.,0.,1.]],'n_valid_mask':{'path':str(path)},'failure_reasons':[]}
            source={'source_size':[10,8],'n_size':[6,5]}
            poly={'source_id':'synthetic','group_id':'synthetic','polygon_id':'synthetic_000','class_id':0,
                  'class_name':'Burn','coordinates':[[3.,2.],[5.,2.],[5.,4.],[3.,4.]]}
            row=polygon_mapping(poly,source,mapping,Image.new('L',(10,8),255))
            self.assertTrue(row['support_complete']);self.assertEqual(row['n_coordinates'][0],[1.,1.])
            poly['coordinates']=[[1.,2.],[4.,2.],[4.,4.],[1.,4.]]
            row=polygon_mapping(poly,source,mapping,Image.new('L',(10,8),255))
            self.assertFalse(row['support_complete'])
            self.assertIn('TARGET_NOT_FULLY_INSIDE_FROZEN_N_FRAME',row['failure_reasons'])

    def test_unverified_mapping_has_no_fake_roi(self):
        p={'source_id':'s','group_id':'g','polygon_id':'s_000','class_id':0,'class_name':'Burn',
           'coordinates':[[1.,1.],[4.,1.],[4.,4.]]}
        m={'mapping_verified':False,'matrix_identity_sha256':None,'n_asset_sha256':'a'*64,
           'failure_reasons':['NO_RELIABLE_FROZEN_HISTORICAL_MAPPING']}
        result=polygon_mapping(p,{'source_size':[10,10],'n_size':[10,10]},m,None)
        self.assertIsNone(result['n_roi_xyxy']);self.assertIsNone(result['n_coordinates']);self.assertFalse(result['support_complete'])

    def test_invalid_affine_and_source_support_failure(self):
        for matrix in ([[0.,0.,0.],[0.,0.,0.]],[[1.,0.,float('nan')],[0.,1.,0.]]):
            with self.assertRaises(ValueError):inverse(matrix)
        with tempfile.TemporaryDirectory() as folder:
            valid=Image.new('L',(10,10),255);valid.putpixel((2,2),0);path=Path(folder)/'valid.png';valid.save(path)
            p={'source_id':'s','group_id':'g','polygon_id':'s_000','class_id':0,'class_name':'Burn',
               'coordinates':[[1.,1.],[4.,1.],[4.,4.],[1.,4.]]}
            m={'mapping_verified':True,'matrix_identity_sha256':'b'*64,'n_asset_sha256':'a'*64,
               'source_to_n':IDENTITY,'n_valid_mask':{'path':str(path)},'failure_reasons':[]}
            result=polygon_mapping(p,{'source_size':[10,10],'n_size':[10,10]},m,valid)
            self.assertFalse(result['support_complete']);self.assertEqual(result['source_valid_target_pixels'],15)

    def test_bounded_edit_scope_does_not_relabel_geometric_validity(self):
        with tempfile.TemporaryDirectory() as folder:
            valid=Image.new('L',(10,10),255);path=Path(folder)/'valid.png';valid.save(path)
            p={'source_id':'s','group_id':'g','polygon_id':'s_000','class_id':0,'class_name':'Burn',
               'coordinates':[[1.,1.],[4.,1.],[4.,4.],[1.,4.]]}
            m={'mapping_verified':True,'matrix_identity_sha256':'b'*64,'n_asset_sha256':'a'*64,
               'source_to_n':IDENTITY,'n_valid_mask':{'path':str(path)},'failure_reasons':[]}
            row=polygon_mapping(p,{'source_size':[10,10],'n_size':[10,10]},m,valid,Image.new('L',(10,10),0))
            self.assertTrue(row['support_complete'])
            self.assertEqual(row['bounded_effective_edit_target_pixels'],0)
            self.assertIn('PRIOR_BASE_PROVENANCE',row['bounded_target_edit_provenance_status'])


if __name__=='__main__':unittest.main()
