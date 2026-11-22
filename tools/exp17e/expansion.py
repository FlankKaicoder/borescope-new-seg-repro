"""Join the P379 census, fixed A10 N and saved geometry into a new review queue.

This is evidence inheritance and review preparation, never automatic Data Gate.
No model, fitting, generator, sampler or historical-file mutation is performed.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ['Burn', 'Crack', 'Dent', 'Material missing', 'Tears', 'Tip curl', 'corrosion']
POLICY = {
    'n_scope': 'Exactly one A10-frozen N per source; never replace it with an A9 recommendation.',
    'd_scope': 'Preserve the original D category and asset separately. A D decision on another N is not transported automatically.',
    'inherited_aligned': 'Exact historically reviewed Aligned bytes, same-review local PASS and B confirmation, complete verified target support.',
    'inherited_derived': 'Crop/Bounded retain the fixed A10 N and require hash-bound proof that all used target pixels equal the accepted Aligned parent. Only the Polygon scope transfers.',
    'inherited_raw': 'Explicit inferential regional-semantic inheritance: exact current Raw SHA is the frozen parent of the same reviewed Aligned, same-review Polygon PASS plus B confirmation, A10 final Raw selection, frozen H and full target support. This is not exact-pixel acceptance or a new user J2 decision.',
    'specific_conflicts': 'Same-current-N/parent D questions and A9 ineligible regional proposals remain actionable. AI conflict does not become user rejection.',
    'new_scope': 'P379 minus B545 is retained in full, including mapping/support exceptions. A10 whole-image acceptance alone cannot create regional approval.',
    'geometry': 'Unverified mapping cannot expose a corresponding N ROI or accept a Pair. Full target support is required; no new matrix is fitted.',
    'gate': 'All inherited and new candidates remain NOT_GRANTED for independent Data Gate; no new human acceptance or Formal Trusted Pair is created.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def count(rows):
    return {'polygons': len(rows), 'source_images': len({r['source_id'] for r in rows}),
            'independent_source_groups': len({r['group_id'] for r in rows})}


def classify(*, identity_ok, asset_exists, mapping_verified, support_complete,
             historical_rejected, region_evidence, current_specific_question,
             proposal_ineligible, had_b545):
    """One primary state; history, evidence route and risks remain separate axes."""
    if not identity_ok:
        return 'IDENTITY_UNRESOLVED', 'UNRESOLVED', True, False, True
    if not asset_exists:
        return 'ASSET_MISSING', 'NOT_RECOVERABLE_WITH_CURRENT_ASSETS', True, False, True
    if not mapping_verified:
        return 'MAPPING_UNVERIFIED', 'NEEDS_TECHNICAL_EVIDENCE', True, False, True
    if not support_complete:
        return 'TECHNICAL_SUPPORT_FAILURE', 'NOT_RECOVERABLE_WITH_CURRENT_ASSETS', True, False, True
    if historical_rejected:
        return 'HISTORICAL_USER_REJECTED', 'NEEDS_HUMAN_SEMANTIC_REVIEW', True, True, False
    if current_specific_question or (proposal_ineligible and had_b545):
        return 'CURRENT_TARGETED_REVIEW', 'NEEDS_HUMAN_SEMANTIC_REVIEW', True, True, True
    if proposal_ineligible or not region_evidence:
        return 'NEW_HUMAN_REVIEW_CANDIDATE', 'READY_FOR_HUMAN_REVIEW', True, True, False
    return 'CURRENT_INHERITED', 'READY_FOR_HUMAN_REVIEW', False, False, False


def inheritance_route(p, source, sm, pm, lineage):
    h = p['same_n_version_history']
    if not h['same_review_user_polygon_pass_and_b_confirmed']:
        return None
    kind = p['n_asset_type']
    if kind == 'ALIGNED' and h['exact_current_n_region_acceptance_evidence']:
        return 'EXACT_REVIEWED_ALIGNED_REGION'
    if kind in {'CROP', 'BOUNDED'}:
        evidence = lineage.get(p['polygon_id'])
        if evidence:
            require((evidence['source_id'], evidence['review_id'], evidence['a10_n_sha256'],
                     evidence['matrix_identity_sha256']) ==
                    (p['source_id'], p['n_review_id'], p['n_asset_sha256'], sm['matrix_identity_sha256']),
                    'Pixel-preserving lineage belongs to another N/matrix')
            require(evidence['reviewed_aligned_parent']['sha256'] == h['historical_aligned_sha256'],
                    'Pixel lineage parent differs from historical regional review')
            require(evidence['historical_same_polygon_state'] == 'PASS' and
                    evidence['historical_b_region_review'] == 'CONFIRMED' and
                    evidence['target_rgb_pixels_equal_parent'] and evidence['different_target_rgb_pixels'] == 0,
                    'Target pixel lineage does not preserve the reviewed region')
            return kind + '_EXACT_REVIEWED_TARGET_PIXEL_DERIVATION'
    if kind == 'RAW' and sm['mapping_verified'] and pm['support_complete']:
        evidence = sm['evidence']
        if (evidence.get('historical_raw_sha256') == p['n_asset_sha256'] and
                evidence.get('historical_aligned_sha256') == h['historical_aligned_sha256'] and
                sm.get('historical_raw_to_source') is not None):
            return 'RAW_INFERENTIAL_REGION_INHERITANCE_VIA_FROZEN_H'
    return None


def current_d_question(p, route):
    old = p.get('exp17d_record')
    if not old or old['data_gate_disposition'] != 'TARGETED_EVIDENCE_REVIEW':
        return False
    if old['review_id'] != p['n_review_id']:
        return False
    same_parent = old['assets']['aligned']['sha256'] == p['same_n_version_history']['historical_aligned_sha256']
    return bool(same_parent and route)


def verify_polygon_mapping(p, sm, pm):
    require((pm['source_id'], pm['group_id'], pm['polygon_id'], pm['class_id'], pm['class_name'],
             pm['n_asset_sha256'], pm['matrix_identity_sha256']) ==
            (p['source_id'], p['group_id'], p['polygon_id'], p['class_id'], p['class_name'],
             p['n_asset_sha256'], sm['matrix_identity_sha256']),
            'Polygon mapping has a different source/group/class/N or matrix identity')


def build(args):
    require(not args.output.exists(), 'A new Exp17-E expansion directory is required')
    source_input = args.census / 'source_scope.json'; polygon_input = args.census / 'p379_polygon_census.json'
    sources = load(source_input)['sources']; original = load(polygon_input)['polygons']
    maps_input = args.mapping / 'source_mappings.json'; poly_maps_input = args.mapping / 'polygon_mappings.json'
    mapping_summary = load(args.mapping / 'mapping_summary.json')
    census_summary = load(args.census / 'summary.json')
    require(census_summary['output_sha256']['source_scope.json'] == sha(source_input) and
            census_summary['output_sha256']['p379_polygon_census.json'] == sha(polygon_input),
            'Frozen census output identity differs')
    require(mapping_summary['private_source_mappings_sha256'] == sha(maps_input) and
            mapping_summary['private_polygon_mappings_sha256'] == sha(poly_maps_input),
            'Frozen mapping output identity differs')
    sms = {m['source_id']: m for m in load(maps_input)['sources']}
    pms = {m['polygon_id']: m for m in load(poly_maps_input)['polygons']}
    require(len(sources) == len({s['source_id'] for s in sources}) == 379 and
            len(original) == len(pms) == len({p['polygon_id'] for p in original}), 'P379/source mapping identity differs')
    require(mapping_summary['scope_sha256'] == sha(source_input) and
            mapping_summary['polygon_census_sha256'] == sha(polygon_input),
            'Mapping was not computed on this exact census')
    lineage = {}
    lineage_paths = [args.mapping / 'bounded_region_lineage.json', args.mapping / 'crop_region_lineage.json']
    lineage_summary = load(args.mapping / 'region_lineage_summary.json')
    require(lineage_summary['bounded_private_sha256'] == sha(lineage_paths[0]) and
            lineage_summary['crop_private_sha256'] == sha(lineage_paths[1]),
            'Frozen target-pixel lineage identity differs')
    for path in lineage_paths:
        for item in load(path)['records']:
            require(item['polygon_id'] not in lineage, 'Duplicate target lineage')
            lineage[item['polygon_id']] = item
    sources_by_id = {}
    for s in sources:
        sm = sms[s['source_id']]; n = s['authoritative_n']
        require((sm['n_asset_sha256'], sm['review_id'], sm['variant_id'], sm['repair_version']) ==
                (n['sha256'], n['review_id'], n['variant_id'], n['version']), 'Source mapping uses a different N')
        sources_by_id[s['source_id']] = {**s, 'original_path': s['source_image_path'],
                                        'source_mapping': sm, 'authoritative_n': {**n, 'type': n['asset_type']}}
    rows = []
    for p in original:
        source = sources_by_id[p['source_id']]; sm = sms[p['source_id']]; pm = pms[p['polygon_id']]
        verify_polygon_mapping(p, sm, pm)
        require((pm['source_id'], pm['n_asset_sha256']) == (p['source_id'], p['n_asset_sha256']) and
                p['n_asset_sha256'] == source['authoritative_n']['sha256'], 'Polygon mapping uses a different fixed N')
        require(p['group_id'] == source['group_id'] and p['class_id'] == CLASSES.index(p['class_name']),
                'Polygon group or class differs')
        h = p['same_n_version_history']; old = p.get('exp17d_record')
        route = inheritance_route(p, source, sm, pm, lineage)
        rejected = h['historical_polygon_state'] == 'RETRY_REPAIR' or h['final_version_choice'] == 'EXCLUDE'
        proposal = h.get('a9_b_proposal_on_this_review')
        ineligible = bool(proposal and proposal['geometry_status'] == 'INELIGIBLE')
        targeted = current_d_question(p, route)
        state, recovery, action, human, technical = classify(
            identity_ok=not (old and old['identity_errors']), asset_exists=Path(p['n_asset_path']).is_file(),
            mapping_verified=pm['mapping_verified'], support_complete=pm['support_complete'],
            historical_rejected=rejected, region_evidence=bool(route), current_specific_question=targeted,
            proposal_ineligible=ineligible, had_b545=bool(p['b545_record']))
        reasons = list(pm['failure_reasons'])
        if ineligible: reasons.extend('SAME_N_A9_' + x for x in proposal['reasons'])
        if targeted: reasons.append('SAME_CURRENT_N_OR_PARENT_EXP17_D_SPECIFIC_QUESTION')
        if rejected: reasons.append('SAME_REVIEW_HISTORICAL_USER_REJECTION_REQUIRES_NEW_EXPLICIT_DECISION')
        if not route: reasons.append('NO_COMPLETE_CURRENT_N_REGIONAL_INHERITANCE_CHAIN')
        if not p['b545_record']: reasons.append('NEW_P379_POLYGON_OUTSIDE_HISTORICAL_B545')
        if old and not p['historical_b_selected_n_unchanged']:
            reasons.append('A10_SELECTED_N_DIFFERS_FROM_D_OR_A9_SELECTED_ALIGNED_KEEP_OLD_DECISION_SEPARATE')
        flags = list(dict.fromkeys(reasons))
        if route and route.startswith('RAW_'): flags.append('RAW_SEMANTIC_INFERENCE_NOT_EXACT_PIXEL_APPROVAL')
        if p['extra_original_annotation_not_selected_by_a9']: flags.append('EXTRA_ORIGINAL_GT_NOT_DROPPED')
        binding = {'source_id': p['source_id'], 'group_id': p['group_id'], 'polygon_id': p['polygon_id'],
                   'class_id': p['class_id'], 'n_asset_sha256': p['n_asset_sha256'], 'repair_version': p['repair_version'],
                   'review_id': p['n_review_id'], 'variant_id': p['n_variant_id'],
                   'matrix_identity_sha256': sm['matrix_identity_sha256'],
                   'valid_mask_sha256': (sm.get('n_valid_mask') or {}).get('sha256'),
                   'source_coordinates_sha256': p['coordinates_sha256']}
        history = {'current_n_same_review_human': {k: v for k, v in h.items() if k != 'ai_decision_separate'},
                   'other_versions': p['historical_all_versions'], 'a10_final_selection':
                       {k: source[k] for k in ['decision_origin', 'human_input_sha256', 'human_reason', 'human_notes',
                                              'recorded_utc', 'selected_note', 'old_visual_note']},
                   'original_exp17d': old, 'a9_b_original': p['b545_record']}
        rows.append({k: p[k] for k in ['source_id', 'group_id', 'polygon_id', 'class_id', 'class_name',
                    'coordinates', 'coordinates_sha256', 'id_basis', 'derived_id_basis', 'annotation_index',
                    'source_size', 'n_size', 'source_image_sha256', 'annotation_sha256', 'n_asset_sha256',
                    'repair_version', 'a9_selected', 'a9_all', 'extra_original_annotation_not_selected_by_a9']} |
                    {'candidate_id': digest(binding), 'binding_identity': binding, 'asset_type': p['n_asset_type'],
                     'n_review_id': p['n_review_id'], 'n_variant_id': p['n_variant_id'], 'polygon_mapping': pm,
                     'mapping_verified': pm['mapping_verified'], 'support_complete': pm['support_complete'],
                     'primary_status': state, 'recoverability': recovery, 'needs_action': action,
                     'needs_human_review': human, 'needs_technical_evidence': technical,
                     'new_scope_polygon': not bool(p['b545_record']),
                     'historical_d_category': p['exp17d_historical_disposition'],
                     'historical_d_selected_asset_matches_current_n': p['historical_b_selected_n_unchanged'],
                     'version_consistent': True, 'a9_d_version_differs_from_current_a10': bool(old and not p['historical_b_selected_n_unchanged']),
                     'historical_user_rejected': rejected, 'history': history,
                     'ai_advisory': {'role': 'HISTORICAL_AI_ONLY_NOT_USER_DECISION', 'same_review_ai_decision': h['ai_decision_separate']},
                     'evidence_route': route, 'specific_d_question_applies_current_n': targeted,
                     'inheritance_limitations': 'Only this Polygon/current frozen N; no whole-frame/context equivalence, new user decision or Data Gate. Raw route is explicit same-generation inference via frozen H.',
                     'flags': flags, 'unresolved_reasons': list(dict.fromkeys(reasons)),
                     'accepted_region_scope': {'kind': 'THIS_ORIGINAL_POLYGON_ONLY', 'source_coordinates_sha256': p['coordinates_sha256'],
                                               'n_coordinates': pm['n_coordinates'], 'other_polygons_included': False,
                                               'whole_image_empty_label_decision': False},
                     'new_user_decision': 'PENDING', 'new_user_acceptance_created': False,
                     'independent_data_gate': 'NOT_GRANTED', 'formal_trusted_pair': False, 'training_approved': False})
    require(len({p['candidate_id'] for p in rows}) == len(rows), 'Candidate identities collide')
    for s in sources:
        require(len({(p['n_asset_sha256'], p['repair_version']) for p in rows if p['source_id'] == s['source_id']}) == 1,
                'A source mixed N assets or repair versions')
    summary = {'status': 'EXP17_E_A10_379_POLYGON_EXPANSION_COMPLETE_PENDING_HUMAN_AND_DATA_GATE',
               'population': count(rows), 'source_scope': 379, 'source_groups': 359,
               'existing_b545_total': 545, 'existing_b545_inside': count([p for p in rows if not p['new_scope_polygon']]),
               'existing_b545_outside': 545 - sum(not p['new_scope_polygon'] for p in rows),
               'new_scope_polygons': count([p for p in rows if p['new_scope_polygon']]),
               'original_d_subset': {kind: count([p for p in rows if p['historical_d_category'] == kind]) for kind in
                   ['INHERITED_REGION_PAIR_CANDIDATE', 'ASSET_BINDING_ONLY_PENDING', 'TARGETED_EVIDENCE_REVIEW']},
               'primary_states': {kind: count([p for p in rows if p['primary_status'] == kind]) for kind in sorted({p['primary_status'] for p in rows})},
               'recoverability': {kind: count([p for p in rows if p['recoverability'] == kind]) for kind in sorted({p['recoverability'] for p in rows})},
               'needs_human_judgment': count([p for p in rows if p['needs_human_review']]),
               'needs_technical_evidence': count([p for p in rows if p['needs_technical_evidence']]),
               'default_action_queue': count([p for p in rows if p['needs_action']]),
               'mapping_supported': count([p for p in rows if p['mapping_verified'] and p['support_complete']]),
               'inherited_by_evidence_route': {route: count([p for p in rows if p['primary_status'] == 'CURRENT_INHERITED' and p['evidence_route'] == route])
                   for route in sorted({p['evidence_route'] for p in rows if p['primary_status'] == 'CURRENT_INHERITED'})},
               'raw_inferential_inheritance_is_not_exact_pixel_approval': True,
               'old_asset_pending_resolved_by_a10_exact_aligned_selection': count([p for p in rows if p['historical_d_category'] == 'ASSET_BINDING_ONLY_PENDING'
                   and p['historical_d_selected_asset_matches_current_n'] and p['primary_status'] == 'CURRENT_INHERITED']),
               'class_coverage': {c: {'all': count([p for p in rows if p['class_name'] == c]),
                   'inherited': count([p for p in rows if p['class_name'] == c and p['primary_status'] == 'CURRENT_INHERITED']),
                   'new_scope': count([p for p in rows if p['class_name'] == c and p['new_scope_polygon']])} for c in CLASSES},
               'one_fixed_a10_n_per_source': True, 'new_human_acceptances': 0, 'data_gate_approved_polygons': 0,
               'formal_trusted_pairs': 0, 'model_forwards': 0, 'optimizer_updates': 0, 'new_registration': 0,
               'policy': POLICY, 'code_sha256': sha(__file__)}
    args.output.mkdir(parents=True)
    inputs = [source_input, polygon_input, maps_input, poly_maps_input, *lineage_paths]
    input_hashes = {p.name: sha(p) for p in inputs}
    queue = {'schema': 'exp17e_a10_fixed_n_review_queue_v1', 'sources': list(sources_by_id.values()),
             'polygons': rows, 'summary': summary, 'input_sha256': input_hashes, 'policy': POLICY,
             'new_user_acceptances': 0, 'independent_data_gate': 'NOT_GRANTED'}
    write(args.output / 'polygon_review_queue.json', queue)
    fields = ['candidate_id', 'source_id', 'group_id', 'polygon_id', 'class_id', 'class_name', 'n_asset_sha256',
              'asset_type', 'repair_version', 'primary_status', 'recoverability', 'historical_d_category',
              'new_scope_polygon', 'evidence_route', 'mapping_verified', 'support_complete', 'flags']
    with (args.output / 'polygon_review_queue.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: json.dumps(p[k], ensure_ascii=False) if isinstance(p[k], (list, dict)) else p[k]
                         for k in fields} for p in rows)
    with (args.output / 'reason_breakdown.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['reason', 'polygons', 'source_images', 'independent_source_groups']); writer.writeheader()
        writer.writerows({'reason': reason, **count([p for p in rows if reason in p['unresolved_reasons']])}
                         for reason in sorted({reason for p in rows for reason in p['unresolved_reasons']}))
    summary['queue_sha256'] = sha(args.output / 'polygon_review_queue.json')
    summary['input_sha256'] = input_hashes
    write(args.output / 'polygon_recovery_summary.json', summary)
    write(args.public, summary)
    print(json.dumps({'status': summary['status'], 'primary_states': summary['primary_states'],
                      'new_scope': summary['new_scope_polygons'], 'queue_sha256': summary['queue_sha256']}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['census', 'mapping', 'output', 'public']:
        parser.add_argument('--' + name, type=Path, required=True)
    build(parser.parse_args())
