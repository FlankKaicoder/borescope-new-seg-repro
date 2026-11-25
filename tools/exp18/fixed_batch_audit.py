"""Explicit audit of the four SHA-bound TRAIN data packets, never weights."""
import argparse
from pathlib import Path

from .common import write_json_exclusive


def audit_fixed(config, output):
    from .data import build_dataset, load_fixed_cpu_packet, load_references, native_evidence, compare_historical
    from .protocol import make_protocol
    from .replay import DecodedAssets, prepare_exposure
    context = build_dataset(config, output / 'dataset', horizon=200)
    reference, r5_reference = load_references(config)
    assets, rows = DecodedAssets(), []
    for spec in config['fixed_batches']:
        packet = load_fixed_cpu_packet(spec)
        position = int(spec['batch_position'])
        batch = packet['batch']
        evidence = native_evidence(context, batch, epoch=0, batch_index=position - 1)
        compare_historical(evidence, batch['j2_replay'], reference, r5_reference)
        prepared = prepare_exposure(batch, context, make_protocol()['matching'], epoch=0,
                                    batch_index=position - 1, asset_cache=assets, materialize_n=True)
        rows.append({'position': position, 'packet_sha256': spec['sha256'], 'native': evidence,
                     'matching_sha256': prepared.plan.sha256,
                     'candidates': len(prepared.collection.candidates), 'matched_rows': len(prepared.plan.a_rows),
                     'N_replay': list(prepared.replay_records), 'coverage': prepared.coverage,
                     'materialized_N_shape': None if prepared.normal_images is None else list(prepared.normal_images.shape),
                     'real_model_constructions': 0})
        del packet, batch, prepared
    context.postcheck(); assets.postcheck()
    return {'status': 'PASS_FOUR_FIXED_TRAIN_PACKETS', 'positions': rows,
            'CUDA_initializations': 0, 'real_checkpoint_tensor_loads': 0,
            'real_model_forwards': 0, 'VAL_passes': 0, 'TEST_accesses': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from .guards import configure_cpu_environment, CPUExecutionGuard
    configure_cpu_environment(args.output)
    import torch
    torch.set_num_threads(1)
    from .data import load_runtime_config
    with CPUExecutionGuard((args.output,), raw_roots=('/root/autodl-tmp/损伤训练数据集',)) as guard:
        result = audit_fixed(load_runtime_config(), args.output)
        write_json_exclusive(args.output / 'receipt.json', {'result': result, 'guard': guard.receipt()})
    print(result['status'], flush=True)


if __name__ == '__main__': main()
