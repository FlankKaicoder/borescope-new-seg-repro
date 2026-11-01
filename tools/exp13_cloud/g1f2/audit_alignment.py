"""One fixed CPU SIFT/RANSAC audit of frozen files; no network or live imports."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


def offline_guard(event, args):
    if event.startswith(('socket.', 'urllib.', 'http.client.', 'subprocess.')) or event == 'os.system':
        raise RuntimeError('CPU offline alignment forbids network and child processes')


sys.addaudithook(offline_guard)
sys.dont_write_bytecode = True
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / 'fixed_protocol.json').read_text(encoding='utf-8'))
INPUT_ROOT = Path(r'E:\borescope_review_assets\cloud_pilot_g1f_1a_preflight')
INPUTS = {'roi.png': INPUT_ROOT / 'inputs/roi.png',
          'mask_rgba.png': INPUT_ROOT / 'inputs/mask_rgba.png',
          'raw_output.png': INPUT_ROOT / 'live_result/raw_output.png'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False,
                  default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist())
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def save_image(path, image):
    require(not path.exists(), 'Refuse image overwrite')
    image.save(path)


def eligibility(keypoints, allowed):
    # Padded zero boundary makes the image edge a support exclusion too.
    distance = cv2.distanceTransform(np.pad(allowed, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    ids = []
    for i, keypoint in enumerate(keypoints):
        x, y = (int(round(v)) for v in keypoint.pt)
        if 0 <= x < allowed.shape[1] and 0 <= y < allowed.shape[0]:
            if distance[y, x] > max(2.0, 1.5 * keypoint.size):
                ids.append(i)
    return ids


def matches_for(roi_keys, raw_keys, roi_desc, raw_desc, roi_ids, raw_ids):
    if not roi_ids or len(raw_ids) < 2:
        return [], {'initial_knn_queries': 0, 'two_neighbor_queries': 0, 'ratio_passed': 0, 'distinct_pairs': 0}
    neighbors = cv2.BFMatcher(cv2.NORM_L2).knnMatch(roi_desc[roi_ids], raw_desc[raw_ids], k=2)
    passed = []
    for pair in neighbors:
        if len(pair) == 2:
            a, b = pair
            if a.distance < 0.75 * b.distance:
                passed.append((a.distance / b.distance, a.distance, roi_ids[a.queryIdx], raw_ids[a.trainIdx]))
    # Remove SIFT multi-orientation duplicates and many-to-one spatial anchors.
    used_roi, used_raw, pairs = set(), set(), []
    for ratio, distance, ri, gi in sorted(passed):
        rbin = tuple(int(v // 2) for v in roi_keys[ri].pt)
        gbin = tuple(int(v // 2) for v in raw_keys[gi].pt)
        if rbin not in used_roi and gbin not in used_raw:
            used_roi.add(rbin)
            used_raw.add(gbin)
            pairs.append({'roi': list(roi_keys[ri].pt), 'raw': list(raw_keys[gi].pt),
                          'ratio': ratio, 'descriptor_l2': distance})
    return pairs, {'initial_knn_queries': len(neighbors), 'two_neighbor_queries': sum(len(p) == 2 for p in neighbors),
                   'ratio_passed': len(passed), 'distinct_pairs': len(pairs)}


def distribution(points):
    if len(points) == 0:
        return {'bbox_xyxy': None, 'span_xy': [0, 0], 'hull_fraction_below_ui_roi': 0,
                'cells': {}, 'supported_cells': [], 'separated_supported_regions': False}
    points = np.asarray(points, dtype=np.float64)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    bins = {}
    for x, y in points:
        if y < 64:
            key = 'UI'
        else:
            key = f'{min(2, int(x / (320/3)))}:{min(1, int((y-64)/88))}'
        bins.setdefault(key, []).append((float(x), float(y)))
    cells = {k: {'count': len(v), 'centroid_xy': np.mean(v, axis=0).tolist()} for k, v in bins.items()}
    supported = [k for k, v in cells.items() if k != 'UI' and v['count'] >= 3]
    separated = any(np.linalg.norm(np.asarray(cells[a]['centroid_xy'])-cells[b]['centroid_xy']) >= 60
                    for i, a in enumerate(supported) for b in supported[i+1:])
    hull_area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))) if len(points) >= 3 else 0.0
    return {'bbox_xyxy': [*minimum.tolist(), *maximum.tolist()], 'span_xy': (maximum-minimum).tolist(),
            'hull_fraction_below_ui_roi': hull_area/(320*176), 'cells': cells,
            'supported_cells': supported, 'separated_supported_regions': separated}


def project(matrix, points):
    return np.asarray(points) @ matrix[:, :2].T + matrix[:, 2]


def estimate(pairs, kind, raw_size, is_mechanical):
    if len(pairs) < 3:
        return {'status': 'INSUFFICIENT_MATCHES', 'matrix_raw_to_roi': None,
                'inlier_count': 0, 'inlier_ratio': 0, 'numeric_screen_pass': False}, None, None
    source = np.float64([p['raw'] for p in pairs])
    target = np.float64([p['roi'] for p in pairs])
    cv2.setRNGSeed(20261101)
    estimator = cv2.estimateAffinePartial2D if kind == 'similarity' else cv2.estimateAffine2D
    matrix, mask = estimator(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                             maxIters=2000, confidence=0.99, refineIters=10)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return {'status': 'ESTIMATION_UNAVAILABLE', 'matrix_raw_to_roi': None,
                'inlier_count': 0, 'inlier_ratio': 0, 'numeric_screen_pass': False}, None, None
    keep = mask.ravel().astype(bool)
    error = np.linalg.norm(project(matrix, source)-target, axis=1)
    count = int(keep.sum())
    coverage = distribution(target[keep])
    linear = matrix[:, :2]
    u, scales, vt = np.linalg.svd(linear)
    rotation = u @ vt
    angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    # Full interpolation footprint must remain inside original raw pixels.
    inverse = cv2.invertAffineTransform(matrix)
    yy, xx = np.indices((240, 320))
    back = project(inverse, np.column_stack((xx.ravel(), yy.ravel())))
    width, height = raw_size
    valid = ((back[:, 0] >= 1) & (back[:, 0] <= width-2) &
             (back[:, 1] >= 1) & (back[:, 1] <= height-2)).reshape(240, 320)
    overlap = float(valid.mean())
    median = float(np.median(error[keep])) if count else None
    p90 = float(np.percentile(error[keep], 90)) if count else None
    criteria = {'non_ui_fit': is_mechanical, 'at_least_12_distinct_inliers': count >= 12,
                'median_at_most_3px': median is not None and median <= 3,
                'p90_at_most_3px': p90 is not None and p90 <= 3,
                'two_separated_supported_mechanical_regions': coverage['separated_supported_regions'],
                'hull_fraction_at_least_0p05': coverage['hull_fraction_below_ui_roi'] >= .05,
                'span_at_least_80x48': coverage['span_xy'][0] >= 80 and coverage['span_xy'][1] >= 48,
                'valid_overlap_at_least_half': overlap >= .5, 'positive_determinant': np.linalg.det(linear) > 0}
    return {'status': 'ESTIMATED', 'matrix_raw_to_roi': matrix.tolist(), 'inlier_count': count,
            'outlier_count': len(pairs)-count, 'inlier_ratio': count/len(pairs),
            'inlier_median_error_roi_px': median, 'inlier_p90_error_roi_px': p90,
            'all_match_median_error_roi_px': float(np.median(error)),
            'inlier_distribution_roi': coverage, 'all_match_distribution_roi': distribution(target),
            'valid_overlap_pixels': int(valid.sum()), 'valid_overlap_fraction': overlap,
            'raw_to_roi_singular_scales': scales.tolist(),
            'raw_to_roi_column_norms_xy': np.linalg.norm(linear, axis=0).tolist(),
            'scale_anisotropy_ratio': float(scales[0]/scales[-1]) if scales[-1] else None,
            'polar_rotation_degrees': angle, 'determinant': float(np.linalg.det(linear)),
            'translation_roi_xy': matrix[:, 2].tolist(), 'screening_criteria': criteria,
            'numeric_screen_pass': all(criteria.values()),
            'human_edge_cavity_gate': 'PENDING_HUMAN_REVIEW'}, keep, error


def match_view(roi, raw, pairs, keep=None):
    scale = 400 / raw.width
    raw_thumb = raw.resize((400, round(raw.height*scale)))
    canvas = Image.new('RGB', (720, max(240, raw_thumb.height)+42), 'white')
    canvas.paste(roi, (0, 42)); canvas.paste(raw_thumb, (320, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), 'ROI native 320x240 | raw thumbnail; green=inlier red=outlier cyan=unclassified', fill='black')
    for i, pair in enumerate(pairs):
        a = (pair['roi'][0], pair['roi'][1]+42)
        b = (320+pair['raw'][0]*scale, 42+pair['raw'][1]*scale)
        color = 'cyan' if keep is None else ('lime' if keep[i] else 'red')
        draw.line((a,b), fill=color, width=1)
        for x,y in (a,b): draw.ellipse((x-2,y-2,x+2,y+2), outline=color)
    return canvas


def review_sheet(path, panels, heading):
    width, height = 1500, 46+360*math.ceil(len(panels)/3)
    canvas = Image.new('RGB', (width,height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    draw.text((8,10), heading, fill='black', font=font)
    for i,(title,img) in enumerate(panels):
        x,y=(i%3)*500,46+(i//3)*360
        draw.text((x+8,y+6),title,fill='black',font=font)
        view=img.convert('RGB').copy()
        factor=min(480/view.width,318/view.height)
        view=view.resize((round(view.width*factor),round(view.height*factor)),Image.Resampling.LANCZOS)
        canvas.paste(view,(x+(500-view.width)//2,y+32+(318-view.height)//2))
    save_image(path,canvas)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-fixed-once',action='store_true')
    args=parser.parse_args()
    require(args.run_fixed_once,'Explicit fixed offline execution flag required')
    out=args.output.resolve()
    require(out.parent==Path(r'E:\borescope_review_assets') and not out.exists(),'New controlled output directory required')
    hashes={name:sha(path) for name,path in INPUTS.items()}
    require(hashes==PROTOCOL['input_sha256'],'Frozen input SHA256 mismatch: STOP')
    roi=Image.open(INPUTS['roi.png']).convert('RGB')
    raw=Image.open(INPUTS['raw_output.png']).convert('RGB')
    api=Image.open(INPUTS['mask_rgba.png'])
    require(roi.size==(320,240) and raw.size==(1024,1024) and api.mode=='RGBA','Input metadata mismatch')
    binary=np.uint8(np.asarray(api.getchannel('A'))==0)*255
    require(int((binary>0).sum())==2394,'Frozen polygon pixel count mismatch')
    out.mkdir(exist_ok=False)
    write_json(out/'ALIGNMENT_RUN_STARTED.json',{'experiment':'Exp13-G1F-2','utc':datetime.now(timezone.utc).isoformat(),
               'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'fixed_protocol.json'),
               'rule':'One fixed run; never remove marker or reuse directory; no scientific retuning','api_posts':0})
    shutil.copyfile(HERE/'fixed_protocol.json',out/'fixed_protocol.json')
    shutil.copyfile(Path(__file__),out/'audit_alignment.py')
    for name,path in INPUTS.items():
        with path.open('rb') as src,(out/name).open('xb') as dst: shutil.copyfileobj(src,dst)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    sift=cv2.SIFT_create(**PROTOCOL['sift'])
    roi_gray=cv2.cvtColor(np.asarray(roi),cv2.COLOR_RGB2GRAY)
    raw_gray=cv2.cvtColor(np.asarray(raw),cv2.COLOR_RGB2GRAY)
    roi_keys,roi_desc=sift.detectAndCompute(roi_gray,None)
    raw_keys,raw_desc=sift.detectAndCompute(raw_gray,None)
    exclusion=cv2.dilate(binary,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
    full_roi=255-exclusion
    mech_roi=full_roi.copy(); mech_roi[:64]=0
    masks={'full_image_defect_excluded_ui_allowed':full_roi,'mechanical_only_ui_and_defect_excluded':mech_roi}
    results={}
    panels=[('Frozen ROI',roi),('Exact frozen binary polygon',Image.fromarray(binary)),('Native square raw (thumbnail only)',raw)]
    warp_panels=[]
    for name,roi_allowed in masks.items():
        raw_allowed=cv2.resize(roi_allowed,raw.size,interpolation=cv2.INTER_NEAREST)
        save_image(out/(name+'_roi_allowed.png'),Image.fromarray(roi_allowed))
        save_image(out/(name+'_raw_allowed.png'),Image.fromarray(raw_allowed))
        roi_ids=eligibility(roi_keys,roi_allowed)
        raw_ids=eligibility(raw_keys,raw_allowed)
        pairs,counts=matches_for(roi_keys,raw_keys,roi_desc,raw_desc,roi_ids,raw_ids)
        is_mechanical=name.startswith('mechanical')
        report={'native_keypoints_roi':len(roi_keys),'native_keypoints_raw':len(raw_keys),
                'eligible_keypoints_roi':len(roi_ids),'eligible_keypoints_raw':len(raw_ids),**counts,'models':{}}
        view=match_view(roi,raw,pairs)
        save_image(out/(name+'_matches.png'),view)
        panels.append((('Non-UI' if is_mechanical else 'UI allowed')+' ratio matches',view))
        for kind in ('similarity','affine'):
            fit,keep,error=estimate(pairs,kind,raw.size,is_mechanical)
            report['models'][kind]=fit
            view=match_view(roi,raw,pairs,keep)
            save_image(out/(name+'_'+kind+'_ransac.png'),view)
            panels.append((('Non-UI ' if is_mechanical else 'UI allowed ')+kind+' inliers/outliers',view))
            with (out/(name+'_'+kind+'_matches.csv')).open('x',encoding='utf-8',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['roi_x','roi_y','raw_x','raw_y','lowe_ratio','descriptor_l2','ransac_inlier','error_roi_px'])
                for i,pair in enumerate(pairs):
                    writer.writerow([*pair['roi'],*pair['raw'],pair['ratio'],pair['descriptor_l2'],
                                     bool(keep[i]) if keep is not None else '',float(error[i]) if error is not None else ''])
            if fit['numeric_screen_pass']:
                matrix=np.float64(fit['matrix_raw_to_roi'])
                aligned=cv2.warpAffine(np.asarray(raw),matrix,(320,240),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                inverse=cv2.invertAffineTransform(matrix)
                yy,xx=np.indices((240,320))
                back=project(inverse,np.column_stack((xx.ravel(),yy.ravel())))
                valid=((back[:,0]>=1)&(back[:,0]<=raw.width-2)&(back[:,1]>=1)&(back[:,1]<=raw.height-2)).reshape(240,320)
                aligned_image=Image.fromarray(aligned)
                validity=Image.fromarray(np.uint8(valid)*255)
                # RGBA explicitly distinguishes invalid support from real dark cavities.
                rgba=aligned_image.convert('RGBA');rgba.putalpha(validity)
                save_image(out/(kind+'_aligned_rgba.png'),rgba)
                save_image(out/(kind+'_validity_mask.png'),validity)
                display=Image.new('RGB',roi.size,(100,40,140));display.paste(aligned_image,mask=validity)
                overlay=Image.blend(roi,display,.5)
                save_image(out/(kind+'_aligned_observation.png'),display)
                save_image(out/(kind+'_overlay.png'),overlay)
                warp_panels.extend([(kind+' aligned; purple invalid',display),(kind+' 50% overlay',overlay),
                                    (kind+' independent valid support',validity)])
                for region,box in {'corrosion':(32,56,168,120),'metal_edge':(24,96,184,144),'cavity':(0,120,176,240)}.items():
                    a,b=roi.crop(box),display.crop(box)
                    compare=Image.new('RGB',(a.width*2+4,a.height),'white')
                    compare.paste(a,(0,0));compare.paste(b,(a.width+4,0))
                    save_image(out/(kind+'_'+region+'_closeup.png'),compare)
                    warp_panels.append((kind+' '+region+': original | mapped',compare))
            else:
                fit['warp_saved']=False
                fit['warp_not_saved_reason']='Fixed non-UI engineering screen not satisfied; no unsupported correspondence image manufactured.'
        results[name]=report
    mechanical=results['mechanical_only_ui_and_defect_excluded']['models']
    candidate=any(fit['numeric_screen_pass'] for fit in mechanical.values())
    state='GEOMETRY_CANDIDATE' if candidate else 'GEOMETRY_UNVERIFIED'
    review_sheet(out/'feature_review_sheet.png',panels,'G1F-2 fixed SIFT + RANSAC; UI-only evidence cannot pass; no parameter search')
    if warp_panels:
        review_sheet(out/'alignment_review_sheet.png',warp_panels,'G1F-2 numeric candidate only; physical edge/cavity and Pair usability require human Gate')
    else:
        notice=Image.new('RGB',(800,200),'white')
        d=ImageDraw.Draw(notice)
        d.text((12,25),'GEOMETRY_UNVERIFIED: neither fixed non-UI fit met the screen.\nNo Similarity/Affine warp, overlay or false local correspondence produced.\nReview feature_review_sheet.png and native original/raw files.\nCorrosion / physical edge / cavity correspondence remains unverified.',fill='black')
        save_image(out/'alignment_not_supported.png',notice)
    require({name:sha(path) for name,path in INPUTS.items()}==hashes,'Frozen inputs changed')
    summary={'experiment':'Exp13-G1F-2','status':state,'stage_status':'G1F_2_ALIGNMENT_AUDIT_COMPLETE / PENDING_CHATGPT_AND_USER_PAIR_GATE',
             'input_sha256':hashes,'input_sizes_wh':{'roi':[320,240],'raw':[1024,1024]},
             'environment':{'opencv':cv2.__version__,'numpy':np.__version__,'opencv_threads':cv2.getNumThreads(),'opencl':cv2.ocl.useOpenCL()},
             'protocol_sha256':sha(HERE/'fixed_protocol.json'),'code_sha256':sha(Path(__file__)),
             'diagnostics':results,'fit_count':4,'parameter_search':False,
             'script_bug_corrections':['Attempt 1 completed fixed computation/views but failed writing JSON because numpy.bool_ is not serializable. Attempt 2 uses identical scientific parameters and converts NumPy scalars only for JSON; first output directory is retained.'],
             'user_visual_quality_acceptance':'User states G1F-1 pseudo-normal appearance acceptable; separate from geometry and Pair evidence.',
             'physical_edge_correspondence':'PENDING_HUMAN_REVIEW' if candidate else 'UNVERIFIED',
             'cavity_correspondence':'PENDING_HUMAN_REVIEW' if candidate else 'UNVERIFIED',
             'trusted_training_pair_admitted':False,'pixel_pair_use':'PENDING_PAIR_GATE' if candidate else 'NOT_SUPPORTED',
             'regional_representation_candidate_retained':True,'api_posts':0,'gpu':0,'training':0,'val_test_access':0}
    write_json(out/'alignment_results.json',summary)
    with (out/'G1F_2_ALIGNMENT_REPORT.md').open('x',encoding='utf-8',newline='\n') as f:
        f.write('# Exp13-G1F-2 fixed offline geometry audit\n\n'+state+'\n\n')
        f.write('User accepts pseudo-normal appearance; geometry is assessed independently. Native SIFT, L2 Lowe ratio 0.75, RANSAC 3 ROI pixels; two masks x two linear models, no search.\n\n')
        f.write('|Set|Model|Eligible ROI/raw|Ratio/distinct|Inliers|Ratio|Median/P90 ROI px|Numeric screen|\n|---|---|---|---|---:|---:|---|---|\n')
        for name,diag in results.items():
            for kind,fit in diag['models'].items():
                f.write(f"|{name}|{kind}|{diag['eligible_keypoints_roi']}/{diag['eligible_keypoints_raw']}|{diag['ratio_passed']}/{diag['distinct_pairs']}|{fit['inlier_count']}|{fit['inlier_ratio']:.4f}|{fit.get('inlier_median_error_roi_px')}/{fit.get('inlier_p90_error_roi_px')}|{fit['numeric_screen_pass']}|\n")
        f.write('\nFull matrices, errors, spatial coverage and overlap are in alignment_results.json and match CSVs. Raw exclusion uses only a conservative normalized-position prior; it does not establish geometry. Descriptor clearance is approximate. RANSAC errors are in-sample fit errors, not independent validation.\n\n')
        f.write('Warping is permitted only for mechanical fits passing the frozen screen. Missing warp views are intentional if evidence is insufficient. Validity masks are separate; invalid pixels are never treated as restored metal. No nonrigid fit, manual correspondences, API, GPU, VAL/TEST or training. Input hashes unchanged. No training Pair is automatically admitted. STOP.\n')
    write_json(out/'output_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='output_sha256.json'})
    print(json.dumps({'status':state,'diagnostics':results,'api_posts':0},ensure_ascii=False,
                     default=lambda value: value.item() if isinstance(value,np.generic) else value.tolist()))


if __name__=='__main__':
    main()
