"""Offline alignment of sealed historical tables. No Torch/model execution.

All numeric fields retain their original units and scientific scope. Existing
five-epoch snapshots do not reveal the onset of functional BN mismatch.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / 'private_assets/exp17_j2_r3_20261124/analysis_v2'
FORMAL = ROOT / 'results/exp17_j2_seed42_formal/completed_r1'
OUT = ROOT / 'private_assets/exp17_j2_r4d_20261124/temporal'
EPOCHS = list(range(5, 101, 5))
DOMAINS = ('all_trainable', 'model.0', 'model.4', 'model.10', 'model.13', 'model.16')
FOCUS_MODULES = {0, 4, 10, 13, 16}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def num(row, key):
    return float(row[key]) if row[key] != '' else None


def write(name, value):
    with (OUT / name).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    # Refuse a reused output directory; existing historical reports stay intact.
    OUT.mkdir(parents=True, exist_ok=False)
    inventory = {}

    def read(path):
        path = path.resolve()
        inventory[str(path.relative_to(ROOT))] = {
            'sha256_before': sha(path), 'bytes': path.stat().st_size,
        }
        with path.open(encoding='utf8', newline='') as f:
            return list(csv.DictReader(f))

    train = read(R3 / 'training_steps.csv')
    epochs = read(R3 / 'training_epochs.csv')
    gradients = read(R3 / 'gradient_domain_trajectory.csv')
    bn = read(R3 / 'bn_layer_summary.csv')
    conv = read(ROOT / 'results/exp17_j2_r3_bn_drift/conv_layer_changes.csv')
    optim = read(R3 / 'optimizer_bn_associations.csv')
    fixed = read(FORMAL / 'validation_fixed_counts.csv')
    standard = read(FORMAL / 'validation_overall.csv')

    assert [int(r['step']) for r in train] == list(range(1, 2101))
    assert [int(r['epoch']) for r in epochs] == list(range(1, 101))
    assert all(sum(int(r['epoch']) == e for r in train) == 21 for e in range(1, 101))
    ep_index = {int(r['epoch']): r for r in epochs}
    fixed_index = {(int(r['epoch']), r['role']): r for r in fixed}
    std_index = {(int(r['epoch']), r['role']): r for r in standard if r['pass'] == 'standard'}
    assert set(fixed_index) == set(std_index) == {(e, r) for e in EPOCHS for r in ('master', 'ema')}
    assert all(int(r['box_tp']) == int(r['mask_tp']) == 0 for r in fixed)
    assert all(float(r['metrics/mAP50-95(M)']) == float(r['metrics/mAP50-95(B)']) == 0 for r in std_index.values())
    assert all(int(r['detections']) == int(r['box_tp']) + int(r['box_fp']) == int(r['mask_tp']) + int(r['mask_fp']) for r in fixed)

    numeric_loss_error = max(abs(float(r['total_loss']) -
        (float(r['yolo_loss']) + 8 * float(r['supcon_loss']) + 4 * float(r['pair_loss']))) for r in train)
    # Expected FP32 loss summation roundoff; no redefinition of the saved loss.
    assert numeric_loss_error < .001
    assert all(float(r['gradient_norm']) > 10 and 0 < float(r['clip_coefficient']) < 1 for r in train)

    aligned = []
    for e in EPOCHS:
        row = ep_index[e]
        window = [x for x in train if e - 4 <= int(x['epoch']) <= e]
        assert len(window) == 105 and sum(int(x['images']) for x in window) == 3340
        record = {'epoch': e, 'committed_update': e * 21, 'epoch_training': {
            'YOLO_native_mean': num(row, 'yolo_loss_mean'),
            'SupCon8_weighted_mean': 8 * num(row, 'supcon_loss_mean'),
            'Pair4_weighted_mean': 4 * num(row, 'pair_loss_mean'),
            'total_loss_saved_mean': num(row, 'total_loss_mean'),
            'preclip_norm_mean': num(row, 'gradient_norm_mean'),
            'clip_coefficient_median': num(row, 'clip_coefficient_median'),
            'active_hinge_step_mean': num(row, 'hinge_active_anchor_fraction_mean'),
            'qualified_ABN_step_mean': num(row, 'effective_ABN_mean'),
            'N_P3_gradient_norm_step_mean_ALREADY_INCLUDES_PAIR4': num(row, 'N_P3_unscaled_gradient_norm_mean'),
            'LR_group0_mean': num(row, 'lr_group0_mean'),
        }, 'preceding_five_epoch_window': {
            'first_epoch': e - 4, 'last_epoch': e, 'updates': len(window),
            'all_updates_clipped': True,
            'YOLO_native_mean': float(np.mean([float(x['yolo_loss']) for x in window])),
            'SupCon8_weighted_mean': 8 * float(np.mean([float(x['supcon_loss']) for x in window])),
            'Pair4_weighted_mean': 4 * float(np.mean([float(x['pair_loss']) for x in window])),
            'preclip_norm_median': float(np.median([float(x['gradient_norm']) for x in window])),
            'clip_coefficient_median': float(np.median([float(x['clip_coefficient']) for x in window])),
            'qualified_ABN_min': min(int(x['effective_ABN']) for x in window),
            'qualified_ABN_max': max(int(x['effective_ABN']) for x in window),
            'hinge_active_step_mean': float(np.mean([float(x['hinge_active_anchor_fraction']) for x in window])),
        }, 'formal_fixed_conf025': {}, 'formal_standard': {}}
        for role in ('master', 'ema'):
            r = fixed_index[e, role]
            record['formal_fixed_conf025'][role] = {k: int(v) for k, v in r.items() if k not in ('epoch', 'role')}
            q = std_index[e, role]
            record['formal_standard'][role] = {
                'Box_AP50_95': float(q['metrics/mAP50-95(B)']),
                'Mask_AP50_95': float(q['metrics/mAP50-95(M)']),
                'predictions_count': None,
                'count_note': 'Standard-pass prediction count was not saved in this table; fixed-pass count is not substituted.'
            }
        aligned.append(record)

    selected_probes = []
    probe_epoch = []
    fields = ('yolo_norm', 'supcon8_norm', 'pair4_norm', 'main_norm', 'joint_norm',
              'pair4_to_main', 'pair_yolo_cosine', 'pair_supcon_cosine',
              'pair_main_cosine', 'global_clip_coefficient')
    for e in EPOCHS:
        for domain in DOMAINS:
            chosen = [r for r in gradients if int(r['epoch']) == e and r['domain'] == domain]
            assert sorted(int(r['batch_position']) for r in chosen) == [1, 7, 11, 21]
            for r in chosen:
                selected_probes.append({'epoch': e, 'domain': domain,
                    'fixed_epoch0_batch_position': int(r['batch_position']),
                    **{k: num(r, k) for k in fields}})
            ag = {'epoch': e, 'domain': domain, 'probe_count': 4,
                  'positive_Pair4_probes': sum(float(r['pair4_norm']) > 0 for r in chosen),
                  'negative_Pair_main_probes': sum(float(r['pair_main_cosine']) < 0 for r in chosen if r['pair_main_cosine'] != '')}
            for k in fields:
                vals = [num(r, k) for r in chosen if r[k] != '']
                ag[k + '_min'] = min(vals) if vals else None
                ag[k + '_median'] = float(np.median(vals)) if vals else None
                ag[k + '_max'] = max(vals) if vals else None
            probe_epoch.append(ag)

    def focus(name):
        return name.startswith('model.') and int(name.split('.')[1]) in FOCUS_MODULES

    master_rows = [r for r in bn if r['label'].startswith('J2_epoch') and r['role'] == 'MASTER' and focus(r['bn'])]
    focus_bn = sorted({r['bn'] for r in master_rows})
    assert len(focus_bn) == 20 and len(master_rows) == 400
    bn_index = {(int(float(r['epoch'])), r['bn'], r['role']): r for r in bn if r['label'].startswith('J2_epoch') and focus(r['bn'])}
    conv_index = {(int(r['epoch']), r['bn']): r for r in conv if focus(r['bn'])}
    optim_index = {(int(float(r['epoch'])), r['parameter']): r for r in optim if r['label'].startswith('J2_epoch') and focus(r['parameter'])}
    bn_fields = ('running_mean_median', 'running_mean_abs_median', 'running_mean_l2',
                 'running_var_median', 'gamma_abs_median', 'beta_abs_median',
                 'effective_gain_abs_median', 'effective_shift_abs_median',
                 'mean_previous_std_median', 'var_previous_abs_log2_median')
    conv_fields = ('conv_weight_l2_l2_across_channels', 'weight_delta_l2_l2_across_channels',
                   'weight_direction_cosine_median', 'gamma_previous_delta_l2_across_channels',
                   'beta_previous_delta_l2_across_channels')
    optim_fields = ('theta_l2', 'm_l2', 'v_sum', 'lr', 'weight_decay', 'adaptive_l2_proxy', 'current_theta_decay_l2_proxy')
    selected_layers = []
    for e in EPOCHS:
        for name in focus_bn:
            parent = name.removesuffix('.bn')
            op_names = {'Conv_weight': parent + '.conv.weight', 'BN_gamma': name + '.weight', 'BN_beta': name + '.bias'}
            record = {'epoch': e, 'BN': name, 'statistic_scope': 'SAVED_BUFFERS_NOT_ACTUAL_INPUT_MISMATCH',
                      'MASTER': {k: num(bn_index[e, name, 'MASTER'], k) for k in bn_fields},
                      'EMA': {k: num(bn_index[e, name, 'EMA'], k) for k in bn_fields},
                      'MASTER_Conv_and_affine': {k: num(conv_index[e, name], k) for k in conv_fields},
                      'MASTER_saved_AdamW_algebra_only': {}}
            for tag, parameter in op_names.items():
                r = optim_index.get((e, parameter))
                record['MASTER_saved_AdamW_algebra_only'][tag] = {k: num(r, k) for k in optim_fields} if r is not None else None
            selected_layers.append(record)

    nonzero_fixed = {role: [e for e in EPOCHS if int(fixed_index[e, role]['detections']) > 0] for role in ('master', 'ema')}
    zero_fixed = {role: [e for e in EPOCHS if int(fixed_index[e, role]['detections']) == 0] for role in ('master', 'ema')}
    saturation = {role: [e for e in EPOCHS if int(fixed_index[e, role]['detections']) == 154 * 300] for role in ('master', 'ema')}
    descriptor = {
        'scope': 'LOCAL_NUMPY_ALIGNMENT_ONLY_NO_MODEL_OR_AUTOGRAD',
        'epochs': EPOCHS, 'checkpoint_spacing_updates': 105,
        'first_saved_checkpoint_epoch': 5, 'first_formal_failure_observed_epoch': 5,
        'earliest_causal_onset_identifiable': False,
        'fixed_conf025_saturated_epochs': saturation,
        'fixed_conf025_zero_detection_epochs': zero_fixed,
        'fixed_conf025_nonzero_detection_epochs': nonzero_fixed,
        'fixed_conf025_FP_reappearance_after_E65': {role: [e for e in nonzero_fixed[role] if e > 65] for role in ('master', 'ema')},
        'all_40_standard_roles_AP0': True, 'all_40_fixed_roles_TP0': True,
        'all_2100_updates_Clip10_triggered': True,
        'epochs_where_saved_probes_have_positive_Pair4_in_all_24_selected_rows': [
            e for e in EPOCHS if all(r['positive_Pair4_probes'] == 4 for r in probe_epoch if r['epoch'] == e)],
        'E1_and_E100_training': {str(e): aligned[-1]['epoch_training'] if e == 100 else {
            'YOLO_native_mean': num(ep_index[e], 'yolo_loss_mean'),
            'SupCon8_weighted_mean': 8 * num(ep_index[e], 'supcon_loss_mean'),
            'Pair4_weighted_mean': 4 * num(ep_index[e], 'pair_loss_mean'),
            'preclip_norm_mean': num(ep_index[e], 'gradient_norm_mean'),
            'clip_coefficient_median': num(ep_index[e], 'clip_coefficient_median'),
        } for e in (1, 100)},
        'field_semantics': {
            'training_losses': 'Saved native YOLO, 8*raw SupCon, 4*raw Pair. Total retains original saved FP32 value.',
            'historic_probe_gradients': 'Already weighted SupCon8 and Pair4 derived from sealed original Gram matrices; do not weight again. Original probes use AMP, unlike R4-C FP32.',
            'N_P3': 'Already contains Pair4; original logged gradient is unscaled and observed before global parameter clipping.',
            'hinge': 'Mean of per-step anchor fractions, not pooled anchor-weighted fraction or formal class/group loss reduction.',
            'AdamW': 'Algebraic proxy from saved moments. Not a newly executed step and not a component attribution.',
            'running_statistics': 'Saved BN buffers may drift normally. Absolute magnitude or change does not measure input-statistic mismatch.',
        },
        'evidence_gaps': [
            'No connected-statistic historical gradient vectors were saved at epochs5–95.',
            'No historical per-BN actual input activation statistics corresponding to these checkpoint epochs were saved.',
            'Early formal AP0 precedes any available checkpoint-to-checkpoint temporal comparison; early and final failure need not share one cause.',
            'Endpoint R4-C gradient direction cannot be treated as the gradient direction of preceding updates.',
            'Original 80 probes are20 checkpoint times by4 fixed batches, not80 independent temporal observations.',
            'Standard-pass prediction counts are absent from the summary; formal fixed counts apply only at confidence.25.',
        ],
    }
    # Preserve exact source identities and verify that read-only processing changed none.
    for rel, row in inventory.items():
        row['sha256_after'] = sha(ROOT / rel)
        assert row['sha256_before'] == row['sha256_after'], rel
    write('epoch_alignment.json', aligned)
    write('historical_probes_selected.json', selected_probes)
    write('historical_probes_by_epoch_domain.json', probe_epoch)
    write('selected_BN_Conv_AdamW.json', selected_layers)
    write('temporal_summary.json', descriptor)
    write('input_manifest.json', {'files': inventory,
        'script_sha256': sha(Path(__file__)), 'all_inputs_unchanged': True,
        'forward_calls': 0, 'autograd_calls': 0, 'optimizer_updates': 0, 'new_VAL': 0, 'TEST': 0,
        'checks': {'continuous_training_updates': 2100, 'formal_fixed_roles': 40,
                   'formal_standard_roles': 40, 'selected_historical_probe_rows': len(selected_probes),
                   'selected_BN_rows': len(selected_layers), 'weighted_loss_max_abs_roundoff': numeric_loss_error}})
    print(json.dumps({'output': str(OUT), 'summary': descriptor}, indent=2))


def early_boundary_supplement():
    """Additional read-only summary; preserves the first completed alignment."""
    assert OUT.is_dir() and not (OUT / 'early_boundaries.json').exists()
    inputs = [R3 / 'training_steps.csv',
              ROOT / 'results/exp17g_formal_protocol/formal_training_protocol_v1.json',
              ROOT / 'docs/exp17/EXP17_J2_R1_ENGINEERING_HANDOFF.md']
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protocol = json.loads(inputs[1].read_text(encoding='utf8'))
    schedule = protocol['optimization'] if 'optimization' in protocol else None
    # Locate the frozen schedule by its unique explicit condition instead of
    # inferring the boundary from a rounded LR value in a historical table.
    def schedule_nodes(value):
        if isinstance(value, dict):
            if 'warmup_condition' in value:
                yield value
            for item in value.values():
                yield from schedule_nodes(item)
        elif isinstance(value, list):
            for item in value:
                yield from schedule_nodes(item)
    found = list(schedule_nodes(protocol))
    assert len(found) == 1 and found[0]['warmup_condition'] == 'zero_based_iteration < 63'
    assert found[0]['warmup_iterations'] == 63
    with inputs[0].open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    idx = {int(r['step']): r for r in rows}
    selected = [1, 21, 42, 62, 63, 64, 84, 104, 105, 106, 126]
    output = []
    for update in selected:
        row = idx[update]
        output.append({'update': update, 'epoch': int(row['epoch']),
            'batch_index_zero_based': int(row['batch_index']),
            'warmup_from_frozen_protocol': update <= 63,
            'LR': [float(row['lr_group' + str(i)]) for i in range(3)],
            'gradient_norm_preclip': float(row['gradient_norm']),
            'clip_coefficient': float(row['clip_coefficient']),
            'YOLO_native': float(row['yolo_loss']),
            'SupCon8': 8 * float(row['supcon_loss']),
            'Pair4': 4 * float(row['pair_loss']),
            'AMP_before_scale_in_saved_step': float(row['amp_scale'])})
    ranges = []
    for label, first, last in [('warmup', 1, 63), ('postwarmup_before_R1', 64, 105),
                              ('first_continuation_epoch', 106, 126)]:
        part = [idx[k] for k in range(first, last + 1)]
        ranges.append({'label': label, 'updates_inclusive': [first, last],
            'update_count': len(part),
            'YOLO_native_mean': float(np.mean([float(r['yolo_loss']) for r in part])),
            'Pair4_mean': 4 * float(np.mean([float(r['pair_loss']) for r in part])),
            'gradient_preclip_median': float(np.median([float(r['gradient_norm']) for r in part])),
            'clip_coefficient_median': float(np.median([float(r['clip_coefficient']) for r in part]))})
    assert all(sha(ROOT / rel) == value for rel, value in before.items())
    write('early_boundaries.json', {'source_SHA256': before, 'selected_updates': output,
        'descriptive_windows': ranges,
        'first_nonwarmup_update': 64, 'first_R1_continuation_update': 106,
        'historical_E5_provenance': {
            'MASTER_standard_and_fixed': 'Originally executed before R1 repair; post-hoc sealed association accepted in R1, not newly generated direct identity fields.',
            'EMA_standard_and_fixed': 'Executed during R1 AUX5 recovery using restored105-step state; no optimizer update or repeated MASTER pass.',
            'E5_training_updates': 'All105 updates precede the R1 isolation repair.',
            'inference': 'R1 activation isolation repair cannot be the onset of the already-observed E5 MASTER AP0; this does not prove all later functional defects have the same cause.',
        }, 'new_forward': 0, 'new_autograd': 0})
    print(json.dumps({'early_boundaries': str(OUT / 'early_boundaries.json'),
                      'first_nonwarmup_update': 64, 'first_R1_update': 106}))


if __name__ == '__main__':
    if sys.argv[1:] == ['--early-boundaries']:
        early_boundary_supplement()
    else:
        assert not sys.argv[1:]
        main()
