"""Prepare one fixed, text-auditable three-source G2 cohort from Pilot-4.

This CPU utility does not implement API/network calls. It consumes only verified
original image/annotation copies and their package TRAIN evidence.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


EXPECTED_ZIP = 'd9f049bda223f6485644da71839aed670a78b169b7411f722593c32fabe04f18'
EXPECTED_SPLIT = '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'
PROMPT_FILES = {'Crack': 'configs/exp13_cloud/g2/CRACK_RESTORATION_V1.txt',
                'corrosion': 'configs/exp13_cloud/g2/CORROSION_RESTORATION_V1.txt'}
SELECTED = [('crack260', '260', 'Crack'), ('crack470', '470', 'Crack'),
            ('corrosion23', '23', 'corrosion')]
EXCLUDED = {'2': 'Previously completed G1F-1/G1F-2 source; explicitly excluded from G2.'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def mask_for(size, shapes):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        require(shape.get('shape_type', '').lower() == 'polygon', 'Only original polygon labels accepted')
        points = shape.get('points')
        require(isinstance(points, list) and len(points) >= 3, 'Polygon has fewer than three points')
        for x, y in points:
            require(0 <= x < size[0] and 0 <= y < size[1], 'Original polygon out of bounds')
        draw.polygon([tuple(point) for point in points], fill=255)
    count = sum(value > 0 for value in mask.get_flattened_data())
    require(0 < count < size[0]*size[1], 'Polygon mask is empty or full-frame')
    return mask, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, required=True,
                        help='Extracted exp13_cloud_pilot4 directory, not the nested blind package')
    parser.add_argument('--pilot4-zip', type=Path, required=True,
                        help='Original Pilot-4 ZIP used for the expected SHA256 check')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Required to create a new exclusive preparation directory')
    args = parser.parse_args()
    require(args.run, 'Pass --run to create the frozen offline inputs')
    base = args.asset_root.resolve()
    output = args.output_root.resolve()
    require((base/'manifest.json').is_file(), 'Pilot-4 manifest missing')
    require(args.pilot4_zip.is_file() and sha(args.pilot4_zip.read_bytes()) == EXPECTED_ZIP,
            'Pilot-4 ZIP missing or SHA256 mismatch: STOP')
    require(not output.exists(), 'Output already exists; refusing to overwrite')
    manifest_bytes = (base/'manifest.json').read_bytes()
    package = json.loads(manifest_bytes)
    integrity = json.loads((base/'integrity_report.json').read_bytes())
    require(integrity.get('status') == 'PASS' and
            integrity.get('checks', {}).get('frozen_train_membership') == 'PASS_4_OF_4' and
            integrity.get('checks', {}).get('original_json_parse') == 'PASS_4_OF_4' and
            integrity.get('checks', {}).get('polygon_bounds') == 'PASS_ALL_POLYGONS',
            'Pilot-4 package integrity gate failed: STOP')
    require(package['frozen_split']['sha256'] == EXPECTED_SPLIT and
            package['frozen_split']['verification_status'] == 'PASS', 'Frozen split evidence mismatch')
    split_rows = {row['stem']: row for row in read_csv(base/'reports/frozen_train_split_rows.csv')}
    source_rows = {row['source_key']: row for row in read_csv(base/'reports/source_mapping.csv')}
    require(len(SELECTED) == 3 and sum(cls == 'Crack' for _,_,cls in SELECTED) == 2 and
            sum(cls == 'corrosion' for _,_,cls in SELECTED) == 1, 'Invalid fixed cohort')
    prompt_text = {}
    prompt_hash = {}
    for cls, relative in PROMPT_FILES.items():
        path = Path(__file__).resolve().parents[3]/relative
        data = path.read_bytes()
        prompt_text[cls] = data.decode('utf-8')
        require(data.endswith(b'\n') and not data.startswith(b'\xef\xbb\xbf'), 'Prompt encoding must be UTF-8 with one terminal newline')
        prompt_hash[cls] = sha(data)
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources').mkdir()
    private = {'stage': 'G2_THREE_SOURCE_PLAN_FROZEN / PENDING_USER_PAID_AUTHORIZATION',
               'package_zip_sha256': EXPECTED_ZIP, 'package_manifest_sha256': sha(manifest_bytes),
               'frozen_split_manifest_sha256': EXPECTED_SPLIT,
               'budget': {'selected_sources': 3, 'max_generation_posts_total': 3,
                          'max_generation_posts_per_source': 1, 'automatic_retries': 0,
                          'fallbacks': 0, 'prompt_search': 0, 'concurrency': 0,
                          'post_authorized': False, 'generation_posts_this_stage': 0,
                          'separate_targeted_repair_budget_authorized': 0},
               'sources': [], 'excluded_sources': EXCLUDED,
               'selection_note': "corrosion23 has prior G1B/G1C/G1D exposure; retained only because the user's revised scope excludes corrosion2 and the verified package contains no other corrosion TRAIN source. G2 call ledger remains separate from G1B and G1F."}
    public = {'record_type': 'G2_THREE_SOURCE_PREPARATION_NOT_AUTHORIZATION',
              'status': private['stage'], 'frozen_split_manifest_sha256': EXPECTED_SPLIT,
              'pilot4_zip_sha256': EXPECTED_ZIP,
              'cohort_size': 3, 'class_counts': {'Crack': 2, 'corrosion': 1},
              'max_generation_posts_total': 3, 'max_generation_posts_per_source': 1,
              'automatic_retries': 0, 'fallbacks': 0, 'prompt_search': 0,
              'generation_posts_this_stage': 0, 'post_authorized': False,
              'g2_cohort_gate': {'required_candidate_count': 3,
                                 'required_classes_with_at_least_one_trusted_candidate': ['Crack','corrosion'],
                                 'statistical_significance_claim': False,
                                 'training_authorized': False}, 'sources': []}
    prompt_index = {}
    for cls, relative in PROMPT_FILES.items():
        prompt_index[cls] = {'path': relative, 'file_sha256': prompt_hash[cls], 'request_rule': 'UTF-8 exact decoded text; remove only terminal LF for multipart value'}
    for key, source_id, cls in SELECTED:
        source_meta = package['sources'][key]
        train = source_meta['train_membership']
        row = source_rows[key]
        frozen_row = split_rows[source_id]
        require(train['status'] == 'PASS' and train['split'] == 'train' and
                train['split_manifest_sha256'] == EXPECTED_SPLIT, 'Source lacks package TRAIN proof: STOP')
        require(row['source_id'] == source_id and row['class'] == cls and row['split'] == 'train', 'Source mapping identity mismatch')
        require(frozen_row['split'] == 'train' and frozen_row['stem'] == source_id and
                frozen_row['group_id'] == train['group_id'], 'Frozen split row mismatch')
        require(frozen_row['sha256'] == row['image_sha256'] == source_meta['original_image']['sha256'] and
                frozen_row['json_sha256'] == row['json_sha256'] == source_meta['original_annotation']['sha256'], 'TRAIN/source hash chain mismatch')
        require(source_id not in EXCLUDED, 'Explicitly excluded source selected')
        image_name = source_meta['original_image']['export_path'].rsplit('/',1)[-1]
        json_name = source_meta['original_annotation']['export_path'].rsplit('/',1)[-1]
        image_bytes = (base/source_meta['original_image']['export_path']).read_bytes()
        json_bytes = (base/source_meta['original_annotation']['export_path']).read_bytes()
        require(sha(image_bytes) == row['image_sha256'] and sha(json_bytes) == row['json_sha256'], 'Actual source files differ from frozen mapping')
        annotation = json.loads(json_bytes.decode('utf-8-sig'))
        with Image.open(base/source_meta['original_image']['export_path']) as im:
            im.load()
            image_size = im.size
            image_format = im.format
            image_mode = im.mode
        require([annotation.get('imageWidth'), annotation.get('imageHeight')] == list(image_size), 'Image/annotation dimensions disagree')
        require(annotation.get('imagePath') == frozen_row['image_path'], 'Annotation image identity mismatch')
        shapes = annotation.get('shapes', [])
        require(shapes and all(s.get('label') == cls for s in shapes), 'Professional JSON class/source mismatch')
        binary, pixels = mask_for(image_size, shapes)
        alpha = Image.eval(binary, lambda value: 255-value)
        rgba = Image.new('RGBA', image_size, (255,255,255,255))
        rgba.putalpha(alpha)
        (output/'sources'/source_id).mkdir(parents=True, exist_ok=False)
        image_dest = output/'sources'/source_id/image_name
        json_dest = output/'sources'/source_id/'original_annotation.json'
        with image_dest.open('xb') as f: f.write(image_bytes)
        with json_dest.open('xb') as f: f.write(json_bytes)
        binary_path = output/'sources'/source_id/'polygon_mask.png'
        rgba_path = output/'sources'/source_id/'api_rgba_mask.png'
        binary.save(binary_path, format='PNG')
        rgba.save(rgba_path, format='PNG')
        rgba_overlay = Image.open(image_dest).convert('RGBA')
        tint = Image.new('RGBA', image_size, (255,35,35,0))
        tint.putalpha(Image.eval(binary, lambda value: 96 if value else 0))
        overlay = Image.alpha_composite(rgba_overlay,tint).convert('RGB')
        draw = ImageDraw.Draw(overlay)
        for shape in shapes:
            points=[tuple(p) for p in shape['points']]
            draw.line(points+[points[0]],fill=(255,255,0),width=max(1,round(min(image_size)/360)))
        overlay_path=output/'sources'/source_id/'polygon_overlay.png'
        overlay.save(overlay_path)
        roi=[0,0,image_size[0],image_size[1]]
        record={'source_key':key,'source_id':source_id,'class':cls,
                'split':'train','group_id':row['group_id'],'source_status':frozen_row['source_status'],
                'frozen_split_manifest_sha256':EXPECTED_SPLIT,
                'train_row_hashes':{'image_sha256':row['image_sha256'],'json_sha256':row['json_sha256']},
                'source_file_names':{'image':image_name,'annotation':'original_annotation.json'},
                'image_size_wh':list(image_size),'image_format':image_format,'image_mode':image_mode,
                'roi_xyxy_half_open':roi,'roi_semantics':'entire original image; no crop, resize, padding or coordinate transform',
                'polygon_count':len(shapes),'polygon_labels':[s['label'] for s in shapes],
                'polygon_point_counts':[len(s['points']) for s in shapes],
                'polygon_bboxes_xyxy':[[min(p[0] for p in s['points']),min(p[1] for p in s['points']),max(p[0] for p in s['points']),max(p[1] for p in s['points'])] for s in shapes],
                'mask_target_pixels':pixels,'mask_coverage_fraction':pixels/(image_size[0]*image_size[1]),
                'api_mask_alpha_semantics':'polygon target alpha=0 (transparent); outside alpha=255 (opaque)',
                'input_hashes':{'original_image':sha(image_bytes),'original_annotation':sha(json_bytes),
                                'polygon_mask':sha(binary_path.read_bytes()),'api_rgba_mask':sha(rgba_path.read_bytes()),
                                'polygon_overlay':sha(overlay_path.read_bytes())},
                'prompt_id':f'{cls.upper()}_RESTORATION_V1','prompt_file_sha256':prompt_hash[cls],
                'request_budget':{'max_generation_posts':1,'automatic_retries':0,'fallbacks':0,'post_authorized':False},
                'prior_cloud_exposure':'G1B/G1C/G1D corrosion23' if source_id=='23' else 'No prior cloud image edit recorded in G1B-G1F'}
        private['sources'].append(record)
        public['sources'].append({k:record[k] for k in ['source_id','class','split','group_id','image_size_wh','roi_xyxy_half_open','polygon_count','mask_target_pixels','mask_coverage_fraction','prompt_id','prompt_file_sha256','prior_cloud_exposure']})
        public['sources'][-1]['input_hashes'] = {
            'original_image_sha256': record['input_hashes']['original_image'],
            'original_professional_json_sha256': record['input_hashes']['original_annotation'],
            'roi_sha256': record['input_hashes']['original_image'],
            'polygon_mask_sha256': record['input_hashes']['polygon_mask'],
            'api_rgba_mask_sha256': record['input_hashes']['api_rgba_mask'],
            'polygon_overlay_sha256': record['input_hashes']['polygon_overlay'],
        }
    require({r['source_id'] for r in private['sources']} == {'260','470','23'}, 'Cohort identity changed')
    require(sum(r['class']=='Crack' for r in private['sources'])==2 and sum(r['class']=='corrosion' for r in private['sources'])==1, 'Class balance mismatch')
    private['prompt_index']=prompt_index
    public['prompt_index']=prompt_index
    public['source_manifest_sha256']=sha(json.dumps(private,ensure_ascii=False,sort_keys=True).encode('utf-8'))
    write_json(output/'private_three_source_manifest.json',private)
    write_json(output/'g2_three_source_manifest.json',public)
    print(json.dumps({'status':'G2_THREE_SOURCE_INPUTS_FROZEN','output_root':str(output),
                      'zip_sha256':EXPECTED_ZIP,'split_sha256':EXPECTED_SPLIT,
                      'sources':[{'source_id':r['source_id'],'class':r['class'],'size':r['image_size_wh'],'polygons':r['polygon_count'],'mask_pixels':r['mask_target_pixels'],'image_sha256':r['input_hashes']['original_image'],'json_sha256':r['input_hashes']['original_annotation'],'mask_sha256':r['input_hashes']['api_rgba_mask'],'prompt_sha256':r['prompt_file_sha256']} for r in private['sources']],
                      'max_posts':3,'post_authorized':False,'api_posts':0},ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('STOP_G2_PREPARATION: '+str(error),file=sys.stderr)
        raise
