import copy
import tempfile
import unittest
from pathlib import Path

from tools.exp17.engineering.data import TrustedPairIndex, digest, label_identity_rows, validate_records


def fixture_index():
    sources=[];records=[]
    for sid,group,labels in [('a','ga',[0,0]),('b','gb',[0]),('c','ga',[0]),('d','gd',[1]),('e','ge',[0])]:
        native=[{'polygon_id':f'{sid}_{i:03d}','annotation_ordinal':i,'native_label_ordinal':i,'class_id':c,
                 'class_name':['Burn','Crack'][c],'coordinates':[[1.,1.],[3.,1.],[2.,3.]]} for i,c in enumerate(labels)]
        source={'source_id':sid,'group_id':group,'native_labels':native,'a_sha256':digest(['A',sid]),
                'annotation_sha256':digest(['JSON',sid]),'a_path':f'/frozen/train/{sid}.png',
                'native_materialized_image_path':f'/release/dataset/images/base/{sid}.png',
                'native_image_relative':f'images/base/{sid}.png'}
        sources.append(source)
        if sid not in ('a','b'):continue
        for n in native:
            record={'source_id':sid,'group_id':group,'polygon_id':n['polygon_id'],'class_id':n['class_id'],
                'class_name':n['class_name'],'native_label_ordinal':n['native_label_ordinal'],
                'annotation_ordinal':n['annotation_ordinal'],'split':'train','asset_type':'ALIGNED',
                'source_coordinates':n['coordinates'],'n_coordinates':n['coordinates'],'source_size':[8,8],'n_size':[8,8],
                'a_sha256':source['a_sha256'],'annotation_sha256':source['annotation_sha256'],'n_sha256':digest(['N',sid]),
                'n_review_id':digest(['review',sid]),'n_variant_id':sid+'_V1','repair_version':'V1',
                'matrix_identity_sha256':digest(['H',sid]),'source_valid_mask_sha256':digest(['valid',sid]),
                'n_valid_mask_sha256':digest(['nvalid',sid]),'n_path':f'/N/{sid}.png','a_path':source['a_path'],
                'source_valid_mask_path':f'/V/{sid}.png','n_valid_mask_path':f'/VN/{sid}.png',
                'evidence_tier':'EXACT_REVIEWED_ALIGNED_REGION','accepted_region_scope':{'scope':'THIS_POLYGON_ONLY'},
                'mapping_verified':True,'support_complete':True}
            record['record_sha256']=digest(record);records.append(record)
    return TrustedPairIndex(records,sources,native_dataset_root='/runtime/dataset')


def native_row(index,sid,ordinal,row,slot):
    n=index.native_label(sid,ordinal)
    return {'row_index':row,'slot':slot,'source_id':sid,'group_id':index.train[sid]['group_id'],
            'polygon_id':n['polygon_id'],'annotation_ordinal':ordinal,'class_id':n['class_id'],'p3_cells':4,
            'augmentation_sha256':digest(['trace',slot])}


class DataTests(unittest.TestCase):
    def test_original_label_ordinal_is_not_class_or_area_matching(self):
        doc={'imageWidth':10,'imageHeight':10,'shapes':[
            {'shape_type':'polygon','label':'Crack','points':[[1,1],[3,1],[2,3],[1,1]]},
            {'shape_type':'polygon','label':'Burn','points':[[1,1],[3,1],[2,3]]}]}
        rows=label_identity_rows('s',doc)
        self.assertEqual([r['polygon_id'] for r in rows],['s_000','s_001'])
        self.assertEqual([r['class_id'] for r in rows],[1,0])
        self.assertEqual(rows[0]['native_label_line'],'1 0.1000000000 0.1000000000 0.3000000000 0.1000000000 0.2000000000 0.3000000000')

    def test_native_path_requires_explicit_full_binding(self):
        index=fixture_index()
        self.assertEqual(index.source_for_image('/runtime/configs/../dataset/images/base/a.png'),'a')
        with self.assertRaises(ValueError):index.source_for_image('/unrelated/a.png')
        self.assertIsNone(index.lookup('e',0))

    def test_unapproved_asset_type_and_record_mutation_rejected(self):
        index=fixture_index();records=copy.deepcopy(list(index.records));records[0]['asset_type']='RAW'
        with self.assertRaises(ValueError):validate_records(records,list(index.train.values()))
        records=copy.deepcopy(list(index.records));records[0]['group_id']='other'
        with self.assertRaises(ValueError):validate_records(records,list(index.train.values()))

    def test_one_current_n_version_per_source(self):
        index=fixture_index();records=copy.deepcopy(list(index.records));records[1]['n_sha256']=digest('otherN')
        records[1]['record_sha256']=digest({k:v for k,v in records[1].items() if k!='record_sha256'})
        with self.assertRaises(ValueError):validate_records(records,list(index.train.values()))

    def test_manifest_sha_hard_gate_and_runtime_asset_relocation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.json';path.write_text('{}',encoding='utf8')
            with self.assertRaises(ValueError):TrustedPairIndex.from_files(path,path,expected_manifest_sha256='0'*64)
        index=fixture_index();record=index.records[0];index.runtime_assets[record['n_sha256']]='/runtime/N/a.png'
        self.assertEqual(str(index.asset_path(record,'n')).replace('\\','/'),'/runtime/N/a.png')


if __name__=='__main__':unittest.main()
