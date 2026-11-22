"""Independent Exp17-F current-byte and frozen geometry-evidence audit.

Uses standard-library reads only: no image decoding, rasterization, registration,
model import, training or change to any prior evidence.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

CSV_SHA="b0a5f768040539a6c2daaab18b1166f99995d4fcf7d0f968e10bec4de2e476fd"
QUEUE_SHA="1bf154d3f0a565892fad12c5f7e80882424baeee34be289cf77911bb8b1ac363"
SPLIT_SHA="35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
CLASSES=['Burn','Crack','Dent','Material missing','Tears','Tip curl','corrosion']
ACCEPT='ACCEPT_THIS_POLYGON_FOR_J2'
REJECT='REJECT_THIS_POLYGON_FOR_J2'


def require(value, reason):
    if not value:raise ValueError(reason)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


class CurrentFiles:
    def __init__(self):self.rows={};self.categories=Counter()

    def check(self,path,expected,category):
        p=Path(path).resolve(strict=True);key=str(p)
        if key not in self.rows:
            require(p.is_file() and sha(p)==expected,'Current frozen input byte identity differs')
            self.rows[key]={'path':key,'sha256':expected,'bytes':p.stat().st_size,'categories':[]}
        require(self.rows[key]['sha256']==expected,'Same path claimed with different SHA256')
        if category not in self.rows[key]['categories']:self.rows[key]['categories'].append(category)
        self.categories[category]+=1
        return p

    def postcheck(self):
        for path,record in self.rows.items():
            require(Path(path).stat().st_size==record['bytes'] and sha(path)==record['sha256'],
                    'Frozen input changed during independent audit')


def matrix3(h):
    result=[[float(v) for v in row] for row in h]
    if len(result)==2:result.append([0.,0.,1.])
    require(len(result)==3 and all(len(r)==3 and all(math.isfinite(v) for v in r) for r in result)
            and result[2]==[0.,0.,1.],'Malformed frozen affine matrix')
    require(result[0][0]*result[1][1]-result[0][1]*result[1][0]>1e-12,'Noninvertible historical affine')
    return result


def inverse(h):
    h=matrix3(h);a,b,c=h[0];d,e,f=h[1];det=a*e-b*d
    return [[e/det,-b/det,(b*f-e*c)/det],[-d/det,a/det,(d*c-a*f)/det],[0.,0.,1.]]


def close(a,b):
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9)


def project(points,h):
    h=matrix3(h)
    return [[h[0][0]*x+h[0][1]*y+h[0][2],h[1][0]*x+h[1][1]*y+h[1][2]] for x,y in points]


def source_technical(source,mapping,entry,files,a10):
    sid=source['source_id'];rec,audit=entry['record'],entry['audit'];kind=source['asset_type']
    require(rec['source_id']==sid and rec['review_id']==source['review_id'] and rec['variant_id']==source['variant_id'],
            'A9 current-N generation identity differs')
    require(rec['source_sha256']==source['source_image_sha256'] and rec['annotation_sha256']==source['annotation_sha256'],
            'Historical source/annotation identity differs')
    for field,hfield,cat in [('source_image_path','source_image_sha256','original_train_image'),
                             ('annotation_path','annotation_sha256','original_professional_json'),
                             ('selected_asset_path','selected_asset_sha256','selected_a10_n')]:
        files.check(source[field],source[hfield],cat)
    require(source['selected_asset_sha256']==source['published_sha256'],'Published/selected frozen N hashes differ')
    files.check(a10/source['published_path'],source['published_sha256'],'published_a10_n')
    annotation=load(source['annotation_path'])
    require([annotation['imageWidth'],annotation['imageHeight']]==source['source_size'],'Frozen original JSON frame differs')
    require(mapping['source_id']==sid and mapping['review_id']==source['review_id'] and mapping['n_asset_sha256']==source['selected_asset_sha256'],
            'Frozen mapping concerns different current N')
    if not mapping['mapping_verified']:
        require(kind=='RAW' and not audit.get('mapping_ok') and mapping['source_to_n'] is None and mapping['n_to_source'] is None,
                'Unverified mapping acquired artificial correspondence')
        return {'source_id':sid,'mapping_verified':False,'asset_type':kind}
    require(audit['mapping_ok'] and rec['geometry_status']=='GEOMETRY_CANDIDATE','Frozen H is not an audited geometric candidate')
    raw_to_source=matrix3(audit['matrix']);require(close(raw_to_source,mapping['historical_raw_to_source']),'Historical H changed')
    for key in ['source_valid_mask','n_valid_mask']:
        desc=mapping[key];files.check(desc['path'],desc['sha256'],key)
    geometry=mapping['evidence'].get('geometry_file')
    if geometry:files.check(geometry['path'],geometry['sha256'],'historical_geometry_file')
    require(mapping['source_valid_mask']['sha256']==audit['assets']['valid.png']['sha256'],'Historical Valid Mask changed')
    if kind=='RAW':
        require(source['selected_asset_sha256']==rec['raw_sha256'] and close(mapping['source_to_n'],inverse(raw_to_source))
                and close(mapping['n_to_source'],raw_to_source),'Raw SHA or inverse mapping direction differs')
    elif kind=='ALIGNED':
        require(source['selected_asset_sha256']==rec['aligned_sha256'] and source['source_size']==source['n_size']
                and close(mapping['source_to_n'],[[1,0,0],[0,1,0],[0,0,1]]),'Aligned source-grid mapping differs')
    elif kind=='CROP':
        special=source['specialty'];box=special['crop_box_xyxy'];x0,y0,x1,y1=box
        require(all(type(v) is int for v in box) and special['parent_to_crop_xy']==[-x0,-y0]
                and source['n_size']==[x1-x0,y1-y0] and special['parent']['sha256']==rec['aligned_sha256'],
                'Crop parent/range/translation identity differs')
        files.check(special['parent']['path'],special['parent']['sha256'],'crop_parent_aligned')
        files.check(special['valid_mask']['path'],special['valid_mask']['sha256'],'crop_parent_valid')
        require(close(mapping['source_to_n'],[[1,0,-x0],[0,1,-y0],[0,0,1]]),'Crop local coordinates differ')
    elif kind=='BOUNDED':
        special=source['specialty'];require(source['selected_asset_sha256']==special['bounded']['sha256']
                and special['all_gt_accounted'] and special['preserves_base_outside_effective_edit'],
                'Bounded frozen composite proof differs')
        for name in ['base','edit_mask','valid_mask']:
            desc=special[name];files.check(desc['path'],desc['sha256'],'bounded_'+name)
        desc=audit['assets']['aligned.png'];files.check(desc['path'],desc['sha256'],'bounded_parent_aligned')
        files.check(audit['origin_paths']['request_path'],rec['request_sha256'],'bounded_request')
        require(special['edit_mask']['sha256']==rec['mask_sha256'] and special['request_sha256']==rec['request_sha256']
                and close(mapping['source_to_n'],[[1,0,0],[0,1,0],[0,0,1]]),'Bounded request/edit/source-grid proof differs')
    return {'source_id':sid,'mapping_verified':True,'asset_type':kind,'geometry_provenance':mapping['geometry_provenance']}


def polygon_technical(poly,source,mapping):
    pm=poly['polygon_mapping'];require(pm['source_id']==poly['source_id'] and pm['polygon_id']==poly['polygon_id']
        and pm['n_asset_sha256']==poly['n_asset_sha256']==source['selected_asset_sha256'],'Polygon/N mapping binding differs')
    require(poly['n_review_id']==source['review_id'] and poly['n_variant_id']==source['variant_id']
            and poly['repair_version']==source['repair_version'],'Polygon current N version differs')
    require(poly['mapping_verified']==pm['mapping_verified']==mapping['mapping_verified'],'Polygon/source mapping status differs')
    if pm['mapping_verified']:
        require(close(project(poly['coordinates'],mapping['source_to_n']),pm['n_coordinates']),
                'Stored current-N Polygon coordinates differ from fixed transform')
        require(pm['matrix_identity_sha256']==mapping['matrix_identity_sha256'],'Polygon matrix identity differs')
    else:
        require(pm['n_coordinates'] is None and pm['n_roi_xyxy'] is None,'Unverified Polygon has a misleading paired ROI')
    if poly['support_complete']:
        require(pm['mapping_verified'] and pm['support_complete'] and pm['source_support_complete'] and pm['n_support_complete']
            and pm['source_polygon_pixels']==pm['source_valid_target_pixels']>0
            and pm['n_polygon_pixels']==pm['n_valid_target_pixels']>0 and pm['n_vertices_inside_frame'],
            'Full support claim differs from frozen per-domain target counts')
        require(all(0<=x<source['n_size'][0] and 0<=y<source['n_size'][1] for x,y in pm['n_coordinates']),
                'Accepted mapped Polygon is outside native N frame')
    else:
        require(not pm['support_complete'],'Queue support claim differs from frozen mapping')
    return bool(poly['mapping_verified'] and poly['support_complete'] and poly['version_consistent'])


def run(args):
    started=time.monotonic();files=CurrentFiles();delivery=load(args.e_delivery)
    def delivered(name):
        record=delivery['required_deliverables'][name]
        return files.check(record['path'],record['sha256'],'prior_e_delivered_metadata')
    queue_path=files.check(delivery['final_queue_original'],QUEUE_SHA,'original_e_queue')
    queue=load(queue_path);require(len(queue['polygons'])==576,'Original E Polygon population differs')
    scope_path=delivered('a10_379_source_scope.json');sources=load(scope_path)['sources']
    census_path=delivered('p379_polygon_census.json');census=load(census_path)['polygons']
    maps_root=Path(delivery['mapping_proofs_directory']);inputs=queue['input_sha256']
    mapping_path=files.check(maps_root/'source_mappings.json',inputs['source_mappings.json'],'frozen_e_source_mapping')
    polygon_path=files.check(maps_root/'polygon_mappings.json',inputs['polygon_mappings.json'],'frozen_e_polygon_mapping')
    source_maps={r['source_id']:r for r in load(mapping_path)['sources']};poly_maps={r['polygon_id']:r for r in load(polygon_path)['polygons']}
    lineages={}
    for name in ['crop_region_lineage.json','bounded_region_lineage.json']:
        path=files.check(maps_root/name,inputs[name],'frozen_target_pixel_lineage')
        for row in load(path)['records']:lineages[row['polygon_id']]=row
    mapping_summary=load(maps_root/'mapping_summary.json')
    a9_path=files.check(args.a9/'audits/verified_evidence_reconciled.json',mapping_summary['a9_saved_evidence_sha256'],'a9_saved_verified_metadata')
    a9={r['record']['review_id']:r for r in load(a9_path)['reviews']}
    receipt=load(args.a10/'release/freeze_receipt.json')
    manifest_path=files.check(args.a10/'registry/a_frozen_manifest.csv',receipt['private_file_sha256']['registry/a_frozen_manifest.csv'],'a10_frozen_manifest')
    with manifest_path.open(encoding='utf-8-sig',newline='') as f:frozen={r['source_id']:r for r in csv.DictReader(f)}
    require(len(sources)==len(frozen)==379 and len({r['group_id'] for r in sources})==359,'A10 source/group scope differs')
    by_source={r['source_id']:r for r in sources}
    source_results=[]
    for source in sources:
        original=frozen[source['source_id']]
        require(all(source[k]==original[k] for k in original),'Exp17-E source metadata differs from actual A10 freeze')
        source_results.append(source_technical(source,source_maps[source['source_id']],a9[source['review_id']],files,args.a10))
    files.check(args.user_csv,CSV_SHA,'original_user_csv')
    with args.user_csv.open(encoding='utf-8-sig',newline='') as f:reviews=[{k:json.loads(v) for k,v in r.items()} for r in csv.DictReader(f)]
    require(len(reviews)==len({r['candidate_id'] for r in reviews})==len({(r['source_id'],r['polygon_id']) for r in reviews})==69,
            'User review identity cardinality differs')
    by_id={r['candidate_id']:r for r in queue['polygons']};review_by_id={r['candidate_id']:r for r in reviews}
    accepted=[r for r in reviews if r['decision']==ACCEPT];rejected=[r for r in reviews if r['decision']==REJECT]
    require(len(accepted)==65 and len(rejected)==4 and all(r['reviewer']=='kk' and r['training_approved'] is False
            and r['independent_data_gate']=='NOT_GRANTED' for r in reviews),'User attribution/decision boundary differs')
    require({r['polygon_id'] for r in rejected}=={'13_002','185_005','616_001','628_000'},'Reject scope differs')
    require({r['polygon_id'] for r in accepted if r['source_id']=='185'}=={'185_002','185_003'},'Mixed source185 acceptance changed')
    technical=[]
    for poly in queue['polygons']:
        source=by_source[poly['source_id']];mapping=source_maps[poly['source_id']]
        require(poly['polygon_mapping']==poly_maps[poly['polygon_id']],'Queue changed the underlying E mapping record')
        require(poly['coordinates']==next(p['coordinates'] for p in census if p['polygon_id']==poly['polygon_id']),
                'Queue Polygon differs from original professional census')
        passed=polygon_technical(poly,source,mapping)
        technical.append({'candidate_id':poly['candidate_id'],'polygon_id':poly['polygon_id'],'technical_preconditions':passed})
        review=review_by_id.get(poly['candidate_id'])
        if review:
            require(all(review[k]==poly[k] for k in ['source_id','group_id','polygon_id','class_id','class_name','source_image_sha256',
                'annotation_sha256','n_asset_sha256','asset_type','repair_version','n_review_id','n_variant_id']),
                'Review identity differs from original Queue')
        if review and review['decision']==ACCEPT:
            require(passed and not poly['historical_user_rejected'] and review['accepted_region_scope']['scope']=='THIS_POLYGON_ONLY',
                    'User ACCEPT lacks frozen technical/identity/regional preconditions')
        if poly['primary_status']=='CURRENT_INHERITED':
            h=poly['history']['current_n_same_review_human']
            require(passed and h['historical_polygon_state']=='PASS' and h['b_region_review']=='CONFIRMED'
                    and h['review_id']==source['review_id'] and not poly['historical_user_rejected'],
                    'Inherited regional semantics or current-N identity differs')
            route=poly['evidence_route']
            if route=='EXACT_REVIEWED_ALIGNED_REGION':
                require(poly['asset_type']=='ALIGNED' and h['historical_aligned_sha256']==poly['n_asset_sha256']
                        and h['exact_current_n_region_acceptance_evidence'],'Exact Aligned route has different pixels')
            elif route=='RAW_INFERENTIAL_REGION_INHERITANCE_VIA_FROZEN_H':
                require(poly['asset_type']=='RAW' and source['variant_metadata']['raw_sha256']==poly['n_asset_sha256'],
                        'Raw inference route does not concern selected Raw generation')
            elif route in {'CROP_EXACT_REVIEWED_TARGET_PIXEL_DERIVATION','BOUNDED_EXACT_REVIEWED_TARGET_PIXEL_DERIVATION'}:
                proof=lineages[poly['polygon_id']]
                require(proof['a10_n_sha256']==poly['n_asset_sha256'] and proof['matrix_identity_sha256']==mapping['matrix_identity_sha256']
                    and proof['target_rgb_pixels_equal_parent'] and proof['different_target_rgb_pixels']==0,
                    'Pixel-preserving target lineage differs')
            else:raise ValueError('Unexpected inherited evidence route')
    inherited=[r for r in queue['polygons'] if r['primary_status']=='CURRENT_INHERITED']
    pool=inherited+[by_id[r['candidate_id']] for r in accepted]
    require(len(inherited)==499 and len(pool)==len({r['candidate_id'] for r in pool})==564,'Candidate pool is not the actual499+65 union')
    require(sum(not r['technical_preconditions'] for r in technical)==8,'Frozen technical HOLD count changed')
    train_path=files.check(args.a10/'configs/frozen_split_manifest.csv',SPLIT_SHA,'full_668_train_metadata')
    with train_path.open(encoding='utf-8-sig',newline='') as f:train=[r for r in csv.DictReader(f) if r['split']=='train']
    require(len(train)==668 and len({r['group_id'] for r in train})==632,'B population was narrowed or regrouped')
    b_groups={c:{r['group_id'] for r in train if c in r['labels_present'].split('|')} for c in CLASSES}
    class_coverage={}
    for label in CLASSES:
        rows=[r for r in pool if r['class_name']==label];new=[r for r in accepted if r['class_name']==label]
        rejected_class=[r for r in rejected if r['class_name']==label]
        class_coverage[label]={'pending_data_gate_polygons':len(rows),'independent_source_groups':len({r['group_id'] for r in rows}),
            'evidence_routes':dict(Counter(r['evidence_route'] for r in rows)), 'new_user_accept':len(new),'user_reject':len(rejected_class),
            'technical_hold':sum(not r['mapping_verified'] or not r['support_complete'] for r in queue['polygons'] if r['class_name']==label),
            'full_train_b_groups':len(b_groups[label]),'groups_with_cross_source_b':len({r['group_id'] for r in rows if b_groups[label]-{r['group_id']}})}
    require(all(len({(r['n_asset_sha256'],r['repair_version']) for r in pool if r['source_id']==sid})==1 for sid in {r['source_id'] for r in pool}),
            'One source acquired multiple current N versions')
    files.postcheck()
    write(args.output/'verified_input_files.json',{'files':files.rows,'all_current_sha256_postchecked':True,'image_decode_calls':0})
    write(args.output/'per_polygon_technical_preconditions.json',{'records':technical,'new_user_decisions_created':0})
    summary={'status':'PASS_CURRENT_ASSET_AND_EVIDENCE_PRECONDITIONS_NOT_DATA_GATE','identity_errors':0,
        'queue_sha256':QUEUE_SHA,'user_csv_sha256':CSV_SHA,'source_assets':379,'asset_types':dict(Counter(r['asset_type'] for r in sources)),
        'input_file_categories':dict(files.categories),'verified_unique_files':len(files.rows),'verified_unique_bytes':sum(r['bytes'] for r in files.rows.values()),
        'elapsed_seconds':time.monotonic()-started,'all_current_sha256_postchecked':True,'image_decode_calls':0,'new_raster_calls':0,
        'polygons':576,'technical_hold':8,'accepted_65_technical_pass':65,'rejected_exact_scope_polygons':4,'source185_accepts_preserved':2,
        'historical499_evidence_routes':dict(Counter(r['evidence_route'] for r in inherited)),
        'new_user65_asset_types':dict(Counter(r['asset_type'] for r in accepted)), 'candidate_pool':564,
        'pool_asset_types':dict(Counter(r['asset_type'] for r in pool)),'pool_independent_groups':len({r['group_id'] for r in pool}),
        'pool_sources':len({r['source_id'] for r in pool}),'class_group_coverage':class_coverage,
        'full_train_b_images':668,'full_train_b_independent_groups':632,'one_current_a10_n_per_source':True,
        'input_manifest_sha256':sha(args.output/'verified_input_files.json'),'technical_matrix_sha256':sha(args.output/'per_polygon_technical_preconditions.json'),
        'source_sha256':sha(__file__),'model_forwards':0,'optimizer_updates':0,'new_registration':0,'new_user_approvals':0,
        'independent_data_gate':'NOT_GRANTED','formal_trusted_pairs':0,
        'limitations':['Revalidates immutable E support/provenance metadata and current bytes, without new pixel or geometry experiments.',
           '35 Raw inherited regions remain same-generation frozen-H inference, distinct from3 newly reviewed Raw acceptances.',
           'Five Crop/Bounded lineages apply to exact target pixels, not all context or Backbone receptive fields.',
           'Full TRAIN B availability is population-level; unchanged native batches may lack cross-group same-class B.']}
    write(args.output/'summary.json',summary);write(args.public,summary)
    print(json.dumps(summary,sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['e-delivery','user-csv','a9','a10','output','public']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except Exception as error:
        write(a.output/'stopped.json',{'status':'STOPPED_WITH_EVIDENCE','error_type':type(error).__name__,'reason':str(error),
            'model_forwards':0,'optimizer_updates':0,'old_files_modified':0})
        raise
