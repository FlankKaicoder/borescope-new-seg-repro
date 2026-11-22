"""Metadata-only A/N coverage against the complete frozen TRAIN B population."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ['Burn', 'Crack', 'Dent', 'Material missing', 'Tears', 'Tip curl', 'corrosion']
SPLIT_SHA = '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def b_group_sets(train):
    groups = {c: set() for c in CLASSES}
    for row in train:
        require(row['split'].lower() == 'train', 'Only TRAIN metadata may form B references')
        labels = set(row['labels_present'].split('|'))
        require(labels <= set(CLASSES), 'Frozen class mapping differs')
        for label in labels:
            groups[label].add(row['group_id'])
    return groups


def concentration(rows):
    counts = Counter(p['group_id'] for p in rows)
    n = sum(counts.values())
    hhi = sum((v / n) ** 2 for v in counts.values()) if n else None
    return {'polygons': n, 'independent_groups': len(counts),
            'max_group_polygon_share': max(counts.values()) / n if n else None,
            'group_share_hhi': hhi, 'effective_groups_inverse_hhi': 1 / hhi if hhi else None,
            'group_polygon_count_histogram': dict(sorted(Counter(counts.values()).items()))}


def compute(polygons, train):
    b_groups = b_group_sets(train)
    train_by_source = {r['stem']: r for r in train}
    require(len(train_by_source) == len(train), 'Duplicate TRAIN source identity')
    by_source = defaultdict(set)
    for p in polygons:
        require(p['class_name'] in CLASSES and p['class_id'] == CLASSES.index(p['class_name']),
                'A class ID/name differs')
        require(p['source_id'] in train_by_source and train_by_source[p['source_id']]['group_id'] == p['group_id'],
                'A source/group is not its frozen TRAIN identity')
        require(p['class_name'] in train_by_source[p['source_id']]['labels_present'].split('|'),
                'A class absent from its original TRAIN source')
        by_source[p['source_id']].add((p['n_asset_sha256'], p['repair_version']))
    require(all(len(versions) == 1 for versions in by_source.values()),
            'One source used more than one A10 N asset/version')
    summary = {}
    for label in CLASSES:
        rows = [p for p in polygons if p['class_name'] == label]
        supported = [p for p in rows if p['mapping_verified'] and p['support_complete']]
        possible = [p for p in supported if p['primary_status'] in {
            'CURRENT_INHERITED', 'CURRENT_ASSET_PENDING', 'CURRENT_TARGETED_REVIEW',
            'NEW_A10_SCOPE_POLYGON', 'NEW_HUMAN_REVIEW_CANDIDATE'}]
        eligible = lambda p: bool(b_groups[label] - {p['group_id']})
        available_b = [p for p in rows if eligible(p)]
        ranking = [p for p in possible if eligible(p)]
        inherited = [p for p in ranking if p['primary_status'] == 'CURRENT_INHERITED']
        count = lambda items: {'polygons': len(items), 'independent_source_groups': len({p['group_id'] for p in items})}
        summary[label] = {'a_n_scope': count(rows), 'reliable_n_and_full_target_support': count(supported),
                          'full_train_b_independent_groups': len(b_groups[label]),
                          'a_groups_with_other_group_same_class_b_in_full_train': count(available_b),
                          'potential_a_b_n_ranking_pending_review_and_data_gate': count(ranking),
                          'inherited_a_b_n_candidates_pending_data_gate': count(inherited),
                          'source_concentration': concentration(rows),
                          'asset_types': {kind: count([p for p in rows if p['asset_type'] == kind])
                                          for kind in ['ALIGNED', 'RAW', 'BOUNDED', 'CROP']}}
    asset_types = {}
    for kind in ['ALIGNED', 'RAW', 'BOUNDED', 'CROP']:
        rows = [p for p in polygons if p['asset_type'] == kind]
        valid = [p for p in rows if p['mapping_verified'] and p['support_complete']]
        asset_types[kind] = {'sources': len({p['source_id'] for p in rows}), 'polygons': len(rows),
                             'source_groups': len({p['group_id'] for p in rows}),
                             'reliable_supported_polygons': len(valid),
                             'reliable_supported_groups': len({p['group_id'] for p in valid})}
    return {'schema': 'exp17e_pair_structure_coverage_v1', 'classes': summary, 'asset_types': asset_types,
            'population_concentration': concentration(polygons),
            'one_frozen_n_per_source': True, 'sources': len(by_source),
            'b_reference_scope': 'FULL_ORIGINAL_668_TRAIN_METADATA_NOT_S379',
            'b_reference_train_images': len(train),
            'b_reference_train_independent_groups': len({r['group_id'] for r in train}),
            'sampler_or_batch_changes': False,
            'extra_b_sampling': False, 'formal_trusted_pairs_created': 0,
            'interpretation': 'Population-level structural availability only; B presence in an actual unchanged native batch is not guaranteed. Potential ranking counts remain subject to regional review and independent Data Gate. No new sampling or model execution.',
            'group_counts_overlap_across_classes_and_asset_types': True}


def run(args):
    require(sha(args.train_manifest) == SPLIT_SHA, 'Frozen split metadata identity differs')
    queue = json.loads(args.queue.read_text(encoding='utf8'))
    with args.train_manifest.open(encoding='utf-8-sig', newline='') as stream:
        train = [r for r in csv.DictReader(stream) if r['split'].lower() == 'train']
    require(len(train) == len({r['stem'] for r in train}) == 668, 'Full TRAIN B metadata count differs')
    report = compute(queue['polygons'], train)
    report['queue_sha256'] = sha(args.queue)
    report['frozen_split_sha256'] = SPLIT_SHA
    report['code_sha256'] = sha(__file__)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'pair_structure_coverage.json', report)
    rows = []
    for label, item in report['classes'].items():
        rows.append({'class_name': label, 'class_id': CLASSES.index(label),
                     'a_n_polygons': item['a_n_scope']['polygons'],
                     'a_n_groups': item['a_n_scope']['independent_source_groups'],
                     'reliable_n_groups': item['reliable_n_and_full_target_support']['independent_source_groups'],
                     'full_train_b_groups': item['full_train_b_independent_groups'],
                     'a_groups_with_independent_b': item['a_groups_with_other_group_same_class_b_in_full_train']['independent_source_groups'],
                     'potential_ranking_groups': item['potential_a_b_n_ranking_pending_review_and_data_gate']['independent_source_groups'],
                     'inherited_ranking_groups': item['inherited_a_b_n_candidates_pending_data_gate']['independent_source_groups'],
                     'maximum_group_share': item['source_concentration']['max_group_polygon_share'],
                     'group_hhi': item['source_concentration']['group_share_hhi']})
    with (args.output / 'class_group_coverage.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if args.public_output:
        write(args.public_output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--train-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public-output', type=Path)
    run(parser.parse_args())
