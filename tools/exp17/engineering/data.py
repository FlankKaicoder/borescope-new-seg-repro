"""Immutable 521-Aligned Data Gate release and audited native-label identity lookup."""
import argparse
import csv
import hashlib
import json
import posixpath
import time
from collections import Counter
from pathlib import Path, PurePosixPath

CLASSES=['Burn','Crack','Dent','Material missing','Tears','Tip curl','corrosion']
SPLIT_SHA='35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'
ALLOWED_TIERS={'EXACT_REVIEWED_ALIGNED_REGION','NEW_USER_ACCEPTED_CURRENT_ALIGNED_POLYGON'}
GATE='USER_AUTHORIZED_521_ALIGNED_ENGINEERING_DATA_GATE'


def require(value,reason):
    if not value:raise ValueError(reason)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def path_identity(path):
    # Native manifest entries may contain configs/../dataset. Resolve lexical
    # aliases only; never infer source identity from a basename or move a file.
    return posixpath.normpath(str(path).replace('\\','/'))


class Files:
    def __init__(self):self.files={}

    def check(self,path,expected):
        p=Path(path).resolve(strict=True);key=str(p)
        if key not in self.files:
            require(p.is_file() and sha(p)==expected,'Frozen data byte identity differs')
            self.files[key]={'path':key,'sha256':expected,'bytes':p.stat().st_size}
        require(self.files[key]['sha256']==expected,'One data path claimed with different identities')
        return p

    def postcheck(self):
        require(all(Path(p).stat().st_size==r['bytes'] and sha(p)==r['sha256'] for p,r in self.files.items()),
                'Data changed during immutable release construction')


def label_identity_rows(source_id,document):
    """Call the unchanged historical converter; then bind each exact label line.

    No sorting/matching by area/class is allowed. Its dict order is original JSON
    annotation order; exact byte comparison proves native label ordinal identity.
    """
    from tools.exp14.label_plan import polygon_rows
    converted=polygon_rows(source_id,document['shapes'],(document['imageWidth'],document['imageHeight']))
    result=[]
    for ordinal,(pid,(class_id,line)) in enumerate(converted.items()):
        require(pid==f'{source_id}_{ordinal:03d}','Historical Polygon ordinal changed')
        result.append({'polygon_id':pid,'annotation_ordinal':ordinal,'native_label_ordinal':ordinal,
            'class_id':class_id,'class_name':CLASSES[class_id], 'coordinates':document['shapes'][ordinal]['points'],
            'native_label_line':line,'normalized_native_coordinates':[float(v) for v in line.split()[1:]]})
    return result


def validate_records(records,train_sources,expected_count=None):
    require(expected_count is None or len(records)==expected_count,'Trusted Polygon count changed')
    require(len({r['polygon_id'] for r in records})==len(records),'Duplicate trusted Polygon identity')
    native={s['source_id']:s for s in train_sources}
    grouped={}
    for r in records:
        require(r['asset_type']=='ALIGNED' and r['evidence_tier'] in ALLOWED_TIERS and r['split']=='train',
                'Unapproved asset type/evidence/split entered Aligned pool')
        require(r['accepted_region_scope']['scope']=='THIS_POLYGON_ONLY' and r['mapping_verified'] and r['support_complete'],
                'Trusted scope/mapping/support changed')
        s=native[r['source_id']];n=s['native_labels'][r['native_label_ordinal']]
        require(r['group_id']==s['group_id'] and n['polygon_id']==r['polygon_id'] and n['class_id']==r['class_id']
                and n['coordinates']==r['source_coordinates'] and r['a_sha256']==s['a_sha256']
                and r['annotation_sha256']==s['annotation_sha256'],'Trusted/native label identity differs')
        require(r['source_coordinates']==r['n_coordinates'] and r['source_size']==r['n_size'],
                'Aligned coordinates must retain the verified source grid')
        require(r['class_name']==CLASSES[r['class_id']],'Frozen class identity differs')
        require(r['record_sha256']==digest({k:v for k,v in r.items() if k!='record_sha256'}),
                'Trusted record identity differs')
        signature=(r['n_sha256'],r['n_review_id'],r['n_variant_id'],r['repair_version'])
        require(r['source_id'] not in grouped or grouped[r['source_id']]==signature,'A source acquired multiple N versions')
        grouped[r['source_id']]=signature
    return True


class TrustedPairIndex:
    """Full TRAIN native identities plus the immutable approved A/N subset."""
    def __init__(self,records,train_sources,manifest_sha256='',runtime_assets=None,native_dataset_root=None):
        validate_records(records,train_sources)
        self.records=tuple(records);self.train={r['source_id']:r for r in train_sources}
        self.by_polygon={r['polygon_id']:r for r in records};self.manifest_sha256=manifest_sha256
        self.runtime_assets=dict(runtime_assets or {});self.path_sources={}
        for source in train_sources:
            paths=[source['a_path'],source['native_materialized_image_path']]
            if native_dataset_root is not None:paths.append(str(PurePosixPath(str(native_dataset_root))/source['native_image_relative']))
            for path in paths:
                key=path_identity(path)
                require(key not in self.path_sources or self.path_sources[key]==source['source_id'],'Ambiguous original-image path identity')
                self.path_sources[key]=source['source_id']

    @classmethod
    def from_files(cls,manifest,train_identity,*,expected_manifest_sha256,runtime_assets=None,native_dataset_root=None):
        require(sha(manifest)==expected_manifest_sha256,'Immutable trusted manifest SHA differs')
        payload=load(manifest);train=load(train_identity)
        require(payload['data_gate']==GATE and payload['count']==521 and not payload['formal_training_authorized'],
                'Engineering manifest authorization scope differs')
        require(sha(train_identity)==payload['native_train_identity_sha256'],'Full native identity release differs')
        validate_records(payload['records'],train['sources'],521)
        return cls(payload['records'],train['sources'],expected_manifest_sha256,runtime_assets,native_dataset_root)

    def source_for_image(self,path):
        key=path_identity(path)
        require(key in self.path_sources,'Original image path is not explicitly bound to frozen TRAIN')
        return self.path_sources[key]

    def native_label(self,source_id,label_ordinal):
        require(source_id in self.train and type(label_ordinal) is int,'Unknown frozen source or ordinal')
        labels=self.train[source_id]['native_labels'];require(0<=label_ordinal<len(labels),'Native label ordinal outside annotation')
        return labels[label_ordinal]

    def lookup(self,source_id,label_ordinal):
        return self.by_polygon.get(self.native_label(source_id,label_ordinal)['polygon_id'])

    def validate_native_region(self,row):
        ordinal=row.get('native_label_ordinal',row.get('annotation_ordinal'))
        native=self.native_label(row['source_id'],ordinal)
        require(row['polygon_id']==native['polygon_id'] and row['class_id']==native['class_id']
                and row['group_id']==self.train[row['source_id']]['group_id'],'Augmented native Polygon/class/group was misbound')
        if 'annotation_ordinal' in row:require(row['annotation_ordinal']==native['annotation_ordinal'],'Annotation/native ordinal mismatch')
        return native

    def asset_path(self,record,kind):
        fields={'n':('n_sha256','n_path'),'a':('a_sha256','a_path'),
                'source_valid':('source_valid_mask_sha256','source_valid_mask_path'),
                'n_valid':('n_valid_mask_sha256','n_valid_mask_path')}
        hash_field,path_field=fields[kind]
        return Path(self.runtime_assets.get(record[hash_field],record[path_field]))


def build(args):
    started=time.monotonic();files=Files();fdel=load(args.f_delivery);edel=load(args.e_delivery)
    fpaths={name:files.check(d['path'],d['sha256']) for name,d in fdel['required_deliverables'].items()}
    merged=load(fpaths['p379_576_merged_decisions.json'])['records']
    candidate_rows=read_csv(fpaths['accepted_564_candidate_manifest.csv'])
    tier_rows=read_csv(fpaths['accepted_564_evidence_tiers.csv'])
    matrix_rows=read_csv(fpaths['data_gate_evidence_matrix.csv'])
    require(len(merged)==576 and len(candidate_rows)==len(tier_rows)==564,'Frozen F candidate population changed')
    accepted={r['candidate_id']:r for r in candidate_rows};tiers={r['candidate_id']:r for r in tier_rows}
    matrix={r['candidate_id']:r for r in matrix_rows};scope_desc=edel['required_deliverables']['a10_379_source_scope.json']
    scope=load(files.check(scope_desc['path'],scope_desc['sha256']))['sources'];scope_by_sid={s['source_id']:s for s in scope}
    mapping_root=Path(edel['mapping_proofs_directory']);queue=load(files.check(edel['final_queue_original'],
        edel['required_deliverables']['polygon_review_queue.json']['sha256']))
    mappings=load(files.check(mapping_root/'source_mappings.json',queue['input_sha256']['source_mappings.json']))['sources']
    mappings={m['source_id']:m for m in mappings}
    receipt=load(args.a10_root/'release/freeze_receipt.json');sealed=receipt['private_file_sha256']
    split_path=files.check(args.a10_root/'configs/frozen_split_manifest.csv',SPLIT_SHA)
    train=[r for r in read_csv(split_path) if r['split']=='train']
    inventory=load(files.check(args.a10_root/'configs/train_entry_files.json',sealed['configs/train_entry_files.json']))['baseline']
    original_by_sid={s['source_id']:s for s in inventory}
    require(len(train)==len(inventory)==668 and len({t['group_id'] for t in train})==632,'Main TRAIN population changed')
    annotations={str(Path(s['annotation_path']).parent) for s in scope};images={str(Path(s['source_image_path']).parent) for s in scope}
    require(len(annotations)==len(images)==1,'Original frozen export roots differ')
    annotation_root=Path(next(iter(annotations)));image_root=Path(next(iter(images)))
    native=[]
    for t in train:
        sid=t['stem'];base=original_by_sid[sid];annotation=files.check(annotation_root/t['json_path'],t['json_sha256'])
        original=files.check(image_root/t['image_path'],t['sha256']);doc=load(annotation)
        rows=label_identity_rows(sid,doc);raw=(''.join(r['native_label_line']+'\n' for r in rows)).encode()
        label=files.check(args.a10_root/'dataset'/base['label_path'],t['label_sha256'])
        require(hashlib.sha256(raw).hexdigest()==t['label_sha256'] and label.read_bytes()==raw
                and len(rows)==int(t['instance_count']),'Native label rows do not equal original annotation order')
        require(base['image_sha256']==t['sha256'] and base['label_sha256']==t['label_sha256'] and base['group_id']==t['group_id'],
                'Native materialized TRAIN identity differs')
        native.append({'source_id':sid,'group_id':t['group_id'],'split':'train','a_path':str(original),'a_sha256':t['sha256'],
            'annotation_path':str(annotation),'annotation_sha256':t['json_sha256'],'label_sha256':t['label_sha256'],
            'source_size':[doc['imageWidth'],doc['imageHeight']],'native_image_relative':base['image_path'],
            'native_label_relative':base['label_path'],'native_materialized_image_path':str(args.a10_root/'dataset'/base['image_path']),
            'native_labels':rows})
    require(sum(len(s['native_labels']) for s in native)==1266,'Native instance exposure population changed')
    n_by_sid={s['source_id']:s for s in native};records=[];held=Counter()
    for wrapper in merged:
        p=wrapper['original']
        if not wrapper['candidate_pool_pending_data_gate']:
            held[wrapper['final_state']]+=1;continue
        require(wrapper['candidate_id'] in accepted and tiers[wrapper['candidate_id']]['evidence_tier']==wrapper['evidence_tier'],
                'F manifest/tier/merged record identity differs')
        if p['asset_type']!='ALIGNED':held[p['asset_type']]+=1;continue
        require(wrapper['evidence_tier'] in ALLOWED_TIERS and p['mapping_verified'] and p['support_complete']
                and not p['historical_user_rejected'] and p['version_consistent'],'Unapproved Aligned record reached main pool')
        s=scope_by_sid[p['source_id']];m=mappings[p['source_id']];pm=p['polygon_mapping'];sid=p['source_id']
        require(m['mapping_verified'] and m['n_asset_sha256']==p['n_asset_sha256']==s['selected_asset_sha256']
                and p['n_review_id']==s['review_id'] and p['n_variant_id']==s['variant_id'],'Aligned current N identity differs')
        require(json.loads(accepted[wrapper['candidate_id']]['current_a10_n_identity'])['n_asset_sha256']==p['n_asset_sha256'],
                'F candidate CSV current N differs')
        require(matrix[wrapper['candidate_id']]['n_asset_sha256']==p['n_asset_sha256'],'Data Gate evidence matrix N differs')
        ordinal=int(p['annotation_index']);native_label=n_by_sid[sid]['native_labels'][ordinal]
        require(native_label['polygon_id']==p['polygon_id'] and native_label['coordinates']==p['coordinates'],
                'Data Gate Polygon differs from exact native label row')
        for desc in [{'path':s['selected_asset_path'],'sha256':s['selected_asset_sha256']},m['source_valid_mask'],m['n_valid_mask']]:
            files.check(desc['path'],desc['sha256'])
        new=wrapper['new_user_review'];review=(new['normalized_coordinate_view'] if new else None)
        if wrapper['evidence_tier'].startswith('NEW_USER_'):
            require(review and review['decision']=='ACCEPT_THIS_POLYGON_FOR_J2','New Aligned acceptance is not an actual user decision')
        record={'candidate_id':wrapper['candidate_id'],'source_id':sid,'group_id':p['group_id'],'polygon_id':p['polygon_id'],
            'class_id':p['class_id'],'class_name':p['class_name'],'split':'train','asset_type':'ALIGNED',
            'native_label_ordinal':ordinal,'annotation_ordinal':ordinal,'a_path':n_by_sid[sid]['a_path'],'a_sha256':p['source_image_sha256'],
            'annotation_path':n_by_sid[sid]['annotation_path'],'annotation_sha256':p['annotation_sha256'],
            'native_label_sha256':n_by_sid[sid]['label_sha256'],'native_label_line':native_label['native_label_line'],
            'n_path':s['selected_asset_path'],'n_published_relative':s['published_path'],'n_sha256':p['n_asset_sha256'],
            'n_review_id':p['n_review_id'],'n_variant_id':p['n_variant_id'],'repair_version':p['repair_version'],
            'matrix_identity_sha256':pm['matrix_identity_sha256'],'historical_raw_to_source':m['historical_raw_to_source'],
            'source_to_n':m['source_to_n'],'geometry_provenance':m['geometry_provenance'],
            'source_valid_mask_path':m['source_valid_mask']['path'],'source_valid_mask_sha256':m['source_valid_mask']['sha256'],
            'n_valid_mask_path':m['n_valid_mask']['path'],'n_valid_mask_sha256':m['n_valid_mask']['sha256'],
            'source_size':p['source_size'],'n_size':p['n_size'],'source_coordinates':p['coordinates'],'n_coordinates':pm['n_coordinates'],
            'accepted_region_scope':{'scope':'THIS_POLYGON_ONLY','source_coordinates':p['coordinates'],'n_coordinates':pm['n_coordinates'],
                'source_roi_xyxy':pm['source_roi_xyxy'],'n_roi_xyxy':pm['n_roi_xyxy']},
            'mapping_verified':True,'support_complete':True,'evidence_tier':wrapper['evidence_tier'],
            'evidence_origin':'NEW_USER_ALIGNED_ACCEPT' if new else 'HISTORICAL_EXACT_ALIGNED_REGION',
            'historical_human_trace':p['history']['current_n_same_review_human']['trace'],
            'new_user_review_metadata':{'reviewer':review['reviewer'],'reviewed_at':review['reviewed_at'],
                'user_notes_sha256':digest(review['user_notes'])} if review else None,
            'f_record_sha256':digest(wrapper),'data_gate':GATE,'formal_training_authorized':False}
        record['record_sha256']=digest(record);records.append(record)
    require(Counter(r['evidence_tier'] for r in records)==Counter(EXACT_REVIEWED_ALIGNED_REGION=459,
            NEW_USER_ACCEPTED_CURRENT_ALIGNED_POLYGON=62),'Aligned Data Gate evidence partition changed')
    validate_records(records,native,521);files.postcheck()
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'native_train_identity.json',{'schema':'exp17_native_668_train_identity_v1','sources':native,
        'images':668,'instances':1266,'groups':632,'split_sha256':SPLIT_SHA,'val_test_assets_read':0})
    payload={'schema':'exp17_j2_521_aligned_formal_trusted_pair_manifest_v1','data_gate':GATE,'count':521,
        'records':records,'native_train_identity_sha256':sha(args.output/'native_train_identity.json'),
        'f_delivery_sha256':sha(args.f_delivery),'f_merged_sha256':sha(fpaths['p379_576_merged_decisions.json']),
        'formal_training_authorized':False,'smoke_requires_engineering_and_calibration_pass':True}
    write(args.output/'j2_521_formal_trusted_pair_manifest.json',payload)
    with (args.output/'j2_521_formal_trusted_pair_manifest.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records)
    write(args.output/'input_hashes.json',{'files':files.files,'postchecked':True})
    b_groups={c:{s['group_id'] for s in native if any(r['class_name']==c for r in s['native_labels'])} for c in CLASSES}
    summary={'status':'PASS_521_ALIGNED_DATAGATE_MANIFEST_AND_NATIVE_IDENTITY_RELEASE','trusted_polygons':521,
        'source_images':len({r['source_id'] for r in records}),'independent_groups':len({r['group_id'] for r in records}),
        'evidence_tiers':dict(Counter(r['evidence_tier'] for r in records)),
        'class_coverage':{c:{'polygons':sum(r['class_name']==c for r in records),
            'independent_a_groups':len({r['group_id'] for r in records if r['class_name']==c}),
            'full_train_b_groups':len(b_groups[c]),'a_groups_with_other_group_b':len({r['group_id'] for r in records if r['class_name']==c and b_groups[c]-{r['group_id']}})} for c in CLASSES},
        'polygons_per_source_histogram':dict(Counter(Counter(r['source_id'] for r in records).values())),
        'held_outside_main_manifest':dict(held),'full_train_images':668,'full_train_instances':1266,'full_train_groups':632,
        'all_native_label_rows_verified_by_exact_byte_reconstruction':1266,'one_n_version_per_source':True,
        'manifest_json_sha256':sha(args.output/'j2_521_formal_trusted_pair_manifest.json'),
        'manifest_csv_sha256':sha(args.output/'j2_521_formal_trusted_pair_manifest.csv'),
        'native_train_identity_sha256':sha(args.output/'native_train_identity.json'),'input_hashes_sha256':sha(args.output/'input_hashes.json'),
        'verified_input_files':len(files.files),'verified_unique_bytes':sum(r['bytes'] for r in files.files.values()),
        'elapsed_seconds':time.monotonic()-started,'source_sha256':sha(__file__),'formal_training_authorized':False,
        'new_optimizer_updates':0,'model_forwards':0,'new_image_copies':0,'val_test_assets_read':0}
    write(args.output/'summary.json',summary);args.public.parent.mkdir(parents=True,exist_ok=True);write(args.public,summary)
    print(json.dumps(summary,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['f-delivery','e-delivery','a10-root','output','public']:parser.add_argument('--'+key,type=Path,required=True)
    build(parser.parse_args())
