"""CPU-only contract checks; no credentials, API calls, or production paths."""
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g2_single_executor import (SOURCE_IDS, encode_multipart, file_sha, load_and_validate_inputs,
                               require, validate_bundle, write_json)


def main():
    repo = Path(__file__).resolve().parents[3]
    asset_root = Path(r'E:\borescope_review_assets\cloud_pilot_g2_three_source_20261101_retry1')
    bundle, bundle_sha = validate_bundle()
    assert bundle_sha == file_sha(repo / 'configs/exp13_cloud/g2/execution_bundle.json')
    assert tuple(bundle['source_order']) == SOURCE_IDS
    assert bundle['evaluation_denominators']['new_cloud_source_results'] == ['260', '470']
    assert bundle['evaluation_denominators']['historical_corrosion_targeted_retest_results'] == ['23']
    assert bundle['evaluation_denominators']['g2_execution_and_cost_records'] == ['260', '470', '23']
    assert bundle['evaluation_denominators']['corrosion23_retest_credible_for_mixed_class_gate'] is True
    assert bundle['max_new_g2_posts_total'] == 3

    checked = []
    for source_id in SOURCE_IDS:
        inputs = load_and_validate_inputs(bundle, source_id, asset_root)
        body, content_type = encode_multipart(source_id, bundle['model'], inputs['prompt_wire_bytes'],
                                              inputs['image_path'], inputs['mask_path'])
        assert f'boundary=borescope_g2_{source_id}_frozen_v2' in content_type
        assert body.count(b'Content-Disposition: form-data;') == 5
        assert inputs['prompt_file_sha256'] == bundle['sources'][source_id]['prompt_file_sha256']
        assert inputs['prompt_transport_sha256'] == bundle['sources'][source_id]['prompt_transport_sha256']
        with Image.open(inputs['image_path']) as image:
            assert list(image.size) == bundle['sources'][source_id]['image_size_wh']
        checked.append({'source_id': source_id, 'role': bundle['sources'][source_id]['source_role'],
                        'multipart_bytes': len(body), 'post_count': 0})

    with tempfile.TemporaryDirectory(prefix='g2_executor_offline_') as temp:
        marker = Path(temp) / 'TEST_ONLY_STARTED.json'
        write_json(marker, {'test_fixture_only': True, 'api_post_count': 0})
        before = marker.read_bytes()
        try:
            write_json(marker, {'overwrite': True})
        except FileExistsError:
            pass
        else:
            raise AssertionError('Exclusive marker creation allowed overwrite')
        assert marker.read_bytes() == before

    print(json.dumps({'status': 'G2_OFFLINE_EXECUTOR_TESTS_PASS',
                      'bundle_sha256': bundle_sha, 'sources': checked,
                      'marker_exclusive_creation': True, 'api_post_count': 0,
                      'network_request_count': 0, 'production_started_marker_created': False}, indent=2))


if __name__ == '__main__':
    main()
