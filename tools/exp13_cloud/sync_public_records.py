"""Publish explicit, reviewed code/text snapshots and hash-only private evidence.

No image decoding, experiment execution, credential access, or networking.
Existing files are never overwritten. Runtime scripts are copied, never imported.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


CODE = {
    'g1b': ('cloud_pilot_g1b_relay/code', [
        'prepare_inputs.py', 'cloud_repair_once.py', 'gate_b_live_once.py',
        'audit_live_output.py', 'audit_output.py', 'relay_probe.py']),
    'g1c': ('cloud_pilot_g1c_postprocess/code', ['run_g1c_postprocess.py', 'finalize_manifest.py']),
    'g1d': ('cloud_pilot_g1d_protocol_audit', ['offline_protocol_audit.py']),
    'g1e': ('cloud_pilot_g1e_mask_control/code', ['prepare_g1e.py', 'g1e_live_pair.py', 'audit_g1e_pair.py']),
    'g1f0': ('cloud_pilot_g1f_roi_prepare/code', ['prepare_g1f0.py']),
    'g1f0b': ('cloud_pilot_g1f_roi_prepare/g1f_0b_4x3', [
        'prepare_roi_4x3.py', 'Start-BorescopeRoiDryRun.ps1', 'verify_dpapi_dry_run.py']),
    'g1f1a': ('cloud_pilot_g1f_1a_preflight', [
        'protocol.py', 'single_repair.py', 'Start-Corrosion2Repair.ps1',
        'audit_saved_output.py', 'prepare_preflight.py', 'offline_protocol_tests.py', 'seal_preflight.py']),
}
PROMPTS = {
    'CLOUD_RESTORATION_V1.txt': 'cloud_pilot_g1b_relay/configs/prompt.txt',
    'MASK_CONTROL_TEST_V1.txt': 'cloud_pilot_g1e_mask_control/configs/prompt_v1.txt',
    'CORROSION2_ROI_REPAIR_V1.txt': 'cloud_pilot_g1f_1a_preflight/prompt_v1.txt',
}
REPORTS = {
    'G1B_summary.md': 'cloud_pilot_g1b_relay/runs/corrosion23_first_call_live/first_call_summary.md',
    'G1C_summary.md': 'cloud_pilot_g1c_postprocess/reports/G1C_FINAL_REPORT.md',
    'G1D_summary.md': 'cloud_pilot_g1d_protocol_audit/G1D_FINAL_REPORT.md',
    'G1E_0_summary.md': 'cloud_pilot_g1e_mask_control/reports/g1e_dry_run_report.md',
    'G1E_1_summary.md': 'cloud_pilot_g1e_mask_control/reports/G1E_FINAL_REPORT.md',
    'G1F_0_summary.md': 'cloud_pilot_g1f_roi_prepare/reports/G1F_0_FINAL_REPORT.md',
    'G1F_0B_summary.md': 'cloud_pilot_g1f_roi_prepare/g1f_0b_4x3/G1F_0B_FINAL_REPORT.md',
    'G1F_1A_summary.md': 'cloud_pilot_g1f_1a_preflight/G1F_1A_PREFLIGHT_REPORT.md',
}
# Only opaque hashes of these private artifacts are published. Never copy bytes.
EVIDENCE = {
    'pilot4.zip': 'exp13_cloud_pilot4_review_assets_20261031.zip',
    'g1b.corrosion23.original': 'cloud_pilot_g1b_relay/runs/corrosion23_first_call_live/original.png',
    'g1b.corrosion23.polygon_mask': 'cloud_pilot_g1b_relay/runs/corrosion23_first_call_live/polygon_mask.png',
    'g1b.corrosion23.api_mask': 'cloud_pilot_g1b_relay/runs/corrosion23_first_call_live/api_mask.png',
    'g1b.corrosion23.raw': 'cloud_pilot_g1b_relay/runs/corrosion23_first_call_live/raw_output.png',
    'g1c.hard': 'cloud_pilot_g1c_postprocess/outputs/composite_hard.png',
    'g1c.feather3': 'cloud_pilot_g1c_postprocess/outputs/composite_feather_inward_3px.png',
    'g1e.A.raw': 'cloud_pilot_g1e_mask_control/runs/condition_A/raw_output.png',
    'g1e.B.raw': 'cloud_pilot_g1e_mask_control/runs/condition_B/raw_output.png',
    'g1f.corrosion2.original': 'cloud_pilot_g1f_roi_prepare/source_snapshot/original_image.png',
    'g1f.corrosion2.annotation': 'cloud_pilot_g1f_roi_prepare/source_snapshot/original_annotation.json',
    'g1f.A.roi': 'cloud_pilot_g1f_roi_prepare/candidates/roi_A/roi_original.png',
    'g1f.B.roi': 'cloud_pilot_g1f_roi_prepare/candidates/roi_B/roi_original.png',
    'g1f.B4x3.roi': 'cloud_pilot_g1f_roi_prepare/g1f_0b_4x3/roi_b_4x3_original.png',
    'g1f.B4x3.binary_mask': 'cloud_pilot_g1f_roi_prepare/g1f_0b_4x3/roi_b_4x3_binary_mask.png',
    'g1f.B4x3.api_mask': 'cloud_pilot_g1f_roi_prepare/g1f_0b_4x3/roi_b_4x3_api_rgba_mask.png',
}
EXPECTED = {
    'g1b.corrosion23.raw': '492ad7779bd026d767e93ff86e059b02f09fbbb5e3be7880bde828acb54903f6',
    'g1e.A.raw': '4deeee9c25fd911833536a12013b4e7200bbabf4a03bc929ef786ac184f9635b',
    'g1e.B.raw': '15d5e03c3728138e4934225d90ec3c289f26d4fd26ba59fcef1457bb3a3d2f4f',
    'g1f.B4x3.roi': '887a206bb4c550f9b8ba006684a93d1089619f06bec57c3d1e6e16648dabd74a',
    'g1f.B4x3.api_mask': 'da23f8bb1533a83f107bf8b2ef5d5d3f072bc0a9f5ba827393b60af107845dfc',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def scan_text(text, label):
    patterns = [r'\bsk-[A-Za-z0-9_-]{16,}', r'\bgh[pousr]_[A-Za-z0-9]{20,}',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                r'01000000d08c9ddf[0-9a-f]{64,}', r'https?://[^\s"<>]+[?&](?:token|sig|key|signature)=']
    if any(re.search(pattern, text, re.I) for pattern in patterns):
        raise ValueError('Suspected sensitive text: ' + label)
    if '\x00' in text:
        raise ValueError('Binary content in public candidate: ' + label)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    args = parser.parse_args()
    assets, repo = args.asset_root.resolve(), args.repo_root.resolve()
    plan = []
    for stage, (folder, names) in CODE.items():
        plan.extend((f'{folder}/{name}', f'tools/exp13_cloud/{stage}/{name}', 'BYTE_IDENTICAL_CODE') for name in names)
    plan.extend((source, f'configs/exp13_cloud/{name}', 'BYTE_IDENTICAL_PROMPT') for name, source in PROMPTS.items())
    plan.extend((source, f'docs/exp13_cloud/reports/{name}', 'ASSET_ROOT_REDACTED_REPORT') for name, source in REPORTS.items())
    # Audit all candidates before any copy. No recursive private-directory copy.
    for source, destination, role in plan:
        p = assets / source
        if not p.is_file() or p.is_symlink() or (repo / destination).exists():
            raise ValueError('Missing source, link, or existing destination: ' + source)
        scan_text(p.read_text(encoding='utf-8-sig'), source)
    evidence = []
    for asset_id, relative in EVIDENCE.items():
        p = assets / relative
        if not p.is_file() or p.is_symlink():
            raise ValueError('Missing explicit private evidence: ' + asset_id)
        value = sha(p)
        if asset_id in EXPECTED and value != EXPECTED[asset_id]:
            raise ValueError('Frozen hash mismatch: ' + asset_id)
        evidence.append({'asset_id': asset_id, 'private_relative_path': relative,
                         'sha256': value, 'size_bytes': p.stat().st_size, 'published_bytes': False})
    provenance = []
    for source, destination, role in plan:
        p, out = assets / source, repo / destination
        out.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha(p)
        if role == 'ASSET_ROOT_REDACTED_REPORT':
            text = p.read_text(encoding='utf-8-sig')
            for root in (str(assets), str(assets).replace('\\', '/')):
                text = text.replace(root, '${PRIVATE_ASSET_ROOT}')
            with out.open('x', encoding='utf-8', newline='\n') as f:
                f.write(text)
        else:
            # Exclusive destination already checked; copy is exact, no code import/run.
            with p.open('rb') as src, out.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        if sha(p) != source_hash:
            raise ValueError('Historical source changed during publication: ' + source)
        provenance.append({'public_path': destination, 'private_source_relative_path': source,
                           'source_sha256': source_hash, 'public_sha256': sha(out), 'transform': role})
    write_csv(repo / 'docs/exp13_cloud/code_provenance.csv', provenance,
              ['public_path', 'private_source_relative_path', 'source_sha256', 'public_sha256', 'transform'])
    write_csv(repo / 'docs/exp13_cloud/evidence_sha256.csv', evidence,
              ['asset_id', 'private_relative_path', 'sha256', 'size_bytes', 'published_bytes'])
    print(json.dumps({'status': 'SAFE_TEXT_SNAPSHOTS_PREPARED', 'code_files': sum(len(v[1]) for v in CODE.values()),
                      'prompts': len(PROMPTS), 'reports': len(REPORTS), 'private_hash_records': len(evidence),
                      'image_bytes_copied': 0, 'api_posts': 0, 'experiment_scripts_executed': 0}))


if __name__ == '__main__':
    main()
