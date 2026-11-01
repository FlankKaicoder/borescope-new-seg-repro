"""CPU-only, additive preparation; no networking or credential access."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent
SOURCE = HISTORY / 'source_snapshot'
ROI = (0, 0, 320, 240)
EXPECTED_IMAGE = '514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03'
EXPECTED_JSON = 'e4fa956d8ae736e4e851dde5fcf66e16faf2bf87deb284c81039da3e5621717a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(name, data):
    with (ROOT / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot():
    return {str(p.relative_to(HISTORY)): sha(p) for p in sorted(HISTORY.rglob('*'))
            if p.is_file() and ROOT not in p.parents}


def save_png(name, image):
    with (ROOT / name).open('xb') as f:
        image.save(f, format='PNG')


def overlay(image, points):
    layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(points, fill=(255, 30, 30, 95), outline=(255, 30, 30, 255), width=2)
    return Image.alpha_composite(image.convert('RGBA'), layer).convert('RGB')


def prepare():
    names = ['history_before_sha256.json', 'roi_b_4x3_original.png',
             'roi_b_4x3_binary_mask.png', 'roi_b_4x3_api_rgba_mask.png',
             'roi_b_4x3_overlay.png', 'roi_b_4x3_manifest.json',
             'roi_b_4x3_review_sheet.png', 'prompt_CLOUD_RESTORATION_V1.txt']
    ensure(not any((ROOT / name).exists() for name in names), 'Refusing overwrite')
    save_json('history_before_sha256.json', snapshot())
    image_path = SOURCE / 'original_image.png'
    json_path = SOURCE / 'original_annotation.json'
    ensure(sha(image_path) == EXPECTED_IMAGE, 'Source image hash mismatch')
    ensure(sha(json_path) == EXPECTED_JSON, 'Source JSON hash mismatch')
    ann = json.loads(json_path.read_text(encoding='utf-8'))
    source = Image.open(image_path).convert('RGB')
    ensure(source.size == (640, 480), 'Wrong full image size')
    ensure((ann['imageWidth'], ann['imageHeight']) == source.size, 'JSON size mismatch')
    ensure(ann['imagePath'] == '2.png', 'Wrong source identity')
    ensure(len(ann['shapes']) == 1, 'Unexpected shape count')
    shape = ann['shapes'][0]
    ensure(shape['label'] == 'corrosion' and shape['shape_type'] == 'polygon', 'Wrong shape')
    ensure(all(float(v).is_integer() for point in shape['points'] for v in point), 'Non-integral coordinates')
    points = [tuple(map(int, point)) for point in shape['points']]
    ensure(len(points) == 26, 'Wrong point count')
    ensure(all(0 <= x < 320 and 0 <= y < 240 for x, y in points), 'Polygon clipped')
    with (SOURCE / 'source_mapping.csv').open(encoding='utf-8-sig', newline='') as f:
        mapping = next(row for row in csv.DictReader(f) if row['source_key'] == 'corrosion2')
    ensure(mapping['split'] == 'train' and mapping['group_id'] == 'singleton_2', 'TRAIN identity mismatch')
    ensure(mapping['image_sha256'] == EXPECTED_IMAGE and mapping['json_sha256'] == EXPECTED_JSON,
           'Frozen mapping hash mismatch')
    full_mask = Image.new('L', source.size, 0)
    ImageDraw.Draw(full_mask).polygon(points, fill=255)
    roi_image = source.crop(ROI)
    mask = full_mask.crop(ROI)
    rgba = Image.new('RGBA', roi_image.size, (255, 255, 255, 255))
    rgba.putalpha(Image.eval(mask, lambda value: 255 - value))
    total = mask.histogram()[255]
    ensure(total == 2394 and full_mask.histogram()[255] == total, 'Defect pixels lost')
    ensure(mask.getbbox() == (40, 64, 160, 113), 'Mask coordinates incorrect')
    preview = overlay(roi_image, points)
    images = {'roi_b_4x3_original.png': roi_image, 'roi_b_4x3_binary_mask.png': mask,
              'roi_b_4x3_api_rgba_mask.png': rgba, 'roi_b_4x3_overlay.png': preview}
    for name, image in images.items():
        save_png(name, image)
    old_b = Image.open(HISTORY / 'candidates/roi_B/roi_original.png').convert('RGB')
    ensure(ImageChops.difference(roi_image.crop((0, 0, 320, 224)), old_b).getbbox() is None,
           'New ROI differs from original B in shared area')
    old_a = Image.open(HISTORY / 'candidates/roi_A/roi_original.png').convert('RGB')
    ensure(ImageChops.difference(source.crop((8, 32, 192, 144)), old_a).getbbox() is None,
           'Original A differs from source crop')

    sheet = Image.new('RGB', (1360, 1180), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)

    def panel(x, y, title, image):
        draw.text((x, y), title, fill='black', font=font)
        sheet.paste(image.convert('RGB'), (x, y + 27))

    panel(12, 8, 'Full original | 640 x 480 | no preprocessing', source)
    full = overlay(source, points)
    d = ImageDraw.Draw(full)
    for rect, color in [((0, 0, 320, 240), 'lime'), ((0, 0, 320, 224), 'magenta'),
                        ((8, 32, 192, 144), 'cyan')]:
        d.rectangle((rect[0], rect[1], rect[2]-1, rect[3]-1), outline=color, width=2)
    panel(696, 8, 'Polygon red | A cyan | B magenta | B4:3 green', full)
    panel(12, 535, 'Old A | 184 x 112', old_a)
    panel(220, 535, 'Old B | 320 x 224', old_b)
    panel(558, 535, 'New B4:3 | 320 x 240', preview)
    panel(900, 535, 'Mask white = defect / alpha 0', mask)
    zoom_box = (24, 48, 184, 132)
    raw_zoom = source.crop(zoom_box).resize((480, 252), Image.Resampling.NEAREST)
    polygon_zoom = overlay(source, points).crop(zoom_box).resize((480, 252), Image.Resampling.NEAREST)
    panel(12, 840, 'True metal/cavity edge | review-only 3x nearest zoom', raw_zoom)
    panel(558, 840, 'Same edge with unchanged JSON polygon | review-only', polygon_zoom)
    draw.text((12, 1130), 'UI remains cropped at x=320. Black cavity is real structure. Polygon touches the metal edge.',
              fill='black', font=font)
    save_png('roi_b_4x3_review_sheet.png', sheet)
    prompt_source = HISTORY / 'configs/prompt_CLOUD_RESTORATION_V1.txt'
    with (ROOT / 'prompt_CLOUD_RESTORATION_V1.txt').open('xb') as f:
        f.write(prompt_source.read_bytes())
    manifest = {
        'experiment': 'Exp13-G1F-0B', 'source': 'corrosion2', 'split': 'train',
        'group_id': 'singleton_2', 'status': 'ROI_4X3_PREPARED_PENDING_DRY_RUN',
        'source_image': {'path': str(image_path), 'sha256': EXPECTED_IMAGE, 'size_wh': [640, 480]},
        'source_json': {'path': str(json_path), 'sha256': EXPECTED_JSON},
        'roi_xyxy_half_open': list(ROI), 'size_wh': [320, 240], 'aspect_ratio': '4:3',
        'resized': False, 'padding_added': False, 'repair_applied': False,
        'polygon_count': 1, 'polygon_point_count': 26, 'polygon_bbox_xyxy_half_open': [40, 64, 160, 113],
        'polygon_complete': True, 'mask_pixels': total, 'mask_coverage_percent': total / 76800 * 100,
        'outside_mask_pixels': 76800 - total,
        'source_to_roi': {'x_roi': 'x_source', 'y_roi': 'y_source', 'scale_x': 1, 'scale_y': 1},
        'roi_to_source': {'x_source': 'x_roi', 'y_source': 'y_roi'},
        'rgba_mask_semantics': {'inside_alpha': 0, 'outside_alpha': 255, 'mode': 'RGBA', 'format': 'PNG'},
        'old_B_shared_area_pixel_identical': True,
        'files': {name: {'sha256': sha(ROOT / name), 'path': str(ROOT / name)} for name in
                  list(images) + ['roi_b_4x3_review_sheet.png', 'prompt_CLOUD_RESTORATION_V1.txt']},
        'risk': {
            'polygon_clipped': False, 'top_ui_truncated': True,
            'normal_metal_context': 'Substantial adjacent undamaged texture is visible; adequacy for a model remains unverified.',
            'real_edge_preserved': True,
            'black_cavity': 'Real gap/occlusion between mechanical surfaces; preserve, never interpret as missing metal.',
            'mask_touches_true_metal_edge': True,
            'edge_overlap_evidence': 'Visual contact along the lower boundary of the upper metal surface near y=110..112; not a segmented edge ground truth.',
            'cavity_fill_risk': 'The target is adjacent to a real cavity. Generative continuation could bridge/fill it or alter the contour.',
            'ui_note': 'The long top time/date string is truncated at x=320. Adding 16 bottom rows does not resolve this.',
            'context_difference': 'A is compact. B4:3 preserves all B pixels plus source rows 224..239, retaining more lower cavity/metal context without changing defect scale.'
        },
        'api_post_count': 0, 'network_request_count': 0, 'gpu': 0, 'training': 0, 'test_access': 0,
    }
    save_json('roi_b_4x3_manifest.json', manifest)
    print('ROI_4X3_PREPARED; mask_pixels=2394; coverage=3.1171875%; API_POST=0')


def finalize():
    before = json.loads((ROOT / 'history_before_sha256.json').read_text(encoding='utf-8'))
    after = snapshot()
    ensure(before == after, 'Historical G1F-0 file content changed')
    save_json('history_integrity_check.json', {'status': 'PASS', 'checked_files': len(before),
              'all_sha256_unchanged': True, 'new_files_outside_this_directory': False,
              'historical_files': before})
    loader = json.loads((ROOT / 'dpapi_loader_result.json').read_text(encoding='utf-8-sig'))
    ensure(loader['status'] == 'DPAPI_LOADER_DRY_RUN_PASS', 'Loader did not pass')
    dry = json.loads(Path(loader['child_report_path']).read_text(encoding='utf-8'))
    ensure(dry['status'] == 'DRY_RUN_PASS' and dry['api_post_count'] == 0, 'Dry run did not pass')
    m = json.loads((ROOT / 'roi_b_4x3_manifest.json').read_text(encoding='utf-8'))
    states = 'G1F_0B_ROI_4X3_PREPARED / DPAPI_LOADER_DRY_RUN_PASS / PENDING_CHATGPT_GATE'
    hashes = '\n'.join(f"| {name} | `{value['sha256']}` |" for name, value in m['files'].items())
    report = f'''# Exp13-G1F-0B 离线准备报告

状态：`{states}`

## 输入与几何

corrosion2 冻结 TRAIN / singleton_2，原图 640×480；原图和 JSON 哈希核对通过。1 个 corrosion polygon、26 点；半开包围框 `[40,64,160,113)`。

ROI_B_4x3 = `[0,0,320,240)`，320×240，4:3。从完整原图直接裁剪，无缩放、拉伸、补边或修复。原图到 ROI 为恒等坐标映射；原 ROI A/B、JSON 与所有 G1F-0 历史文件逐一哈希未变（{len(before)} 个文件）。

缺陷 Mask 为 2394 像素，覆盖率 **3.1171875%**，正常/非目标区 74406 像素。与完整原图 mask 的裁剪逐像素一致，全部 polygon 完整保留。API Mask 为 RGBA PNG，polygon 内 alpha=0、外部 alpha=255；重新打开文件逐像素核对通过。

## 结构风险审查

- 与原 A `[8,32,192,144)` 相比，保留更多相邻正常金属纹理、真实曲面边缘和黑色空腔；未改变缺陷尺度。
- 与原 B `[0,0,320,224)` 相比，仅增加下方 16 行真实原图像素，旧 B 的共享区域逐像素完全相同；更多下方正常金属及空腔上下文被保留。
- 黑色区域是机械表面的间隙/遮挡和空腔，必须保留，不能作为缺失金属填平。
- 目标 polygon 下边界与上方金属表面的真实轮廓相邻/视觉接触（约 y=110..112）。生成时存在抹平真实边缘、桥接/误填空腔的风险；未为规避风险改变 polygon。
- 顶部时间/日期 UI 在 x=320 处仍被裁断。4:3 修正没有消除 UI 风险，不能认定为完全无风险。
- 正常纹理可见且数量充足；这些观察不能证明模型会遵守结构边缘或独立 Mask。

## SHA256

| 文件 | SHA256 |
|---|---|
{hashes}

源图 SHA256：`{EXPECTED_IMAGE}`。

JSON SHA256：`{EXPECTED_JSON}`。

## DPAPI 与 Dry Run

通过 Windows User 环境变量读取 Base URL `https://direct-api.ccccapi.cc`、模型 `gpt-image-2.5-sunburst`。仅读取 DPAPI 文件第一行并按当前 Windows 用户解密，没有密钥参数或密钥输出。密钥只加入固定 Python 子进程的环境，未设置父进程、系统或用户 OPENAI_API_KEY；未改 Codex 认证配置。子进程启动后立即从自己的环境移除密钥，只保留是否存在的布尔值。

启动器默认 Dry Run；`-Live` 在解密前被拒绝。本阶段没有 live 实现。Python 没有 HTTP 客户端或生图代码，并安装了网络/子进程审计拒绝钩子。验证了原图/JSON、320×240 ROI、完整 polygon、透明方向、模型及 multipart 元数据结构。字段沿用 `model,prompt,n,image[],mask`；固定 Prompt 从 G1F-0 按字节复制。size/quality 不额外指定，n=1；未来输出尺寸仍需实际审计。

DPAPI 加载及 Codex 启动的本地 Python 子进程配置读取均 PASS。启动器在 finally 清理 BSTR、SecureString 及临时引用，移除 ProcessStartInfo 中的密钥项；DPAPI 文件哈希未变。托管字符串不能保证物理内存立即清零，当前用户下有权限读取进程内存的程序仍可能观察瞬时明文；未观察到日志、文件、命令行或网络泄露，不宣称绝对零风险。

API POST=0；网络请求=0；自动重试=0；模型 fallback=0；Prompt 搜索=0；GPU=0；训练=0；TEST=0。

详细记录：`dpapi_loader_result.json`、`{Path(loader['child_report_path']).relative_to(ROOT).as_posix()}`。

## 下一阶段条件

先人工审核 `roi_b_4x3_review_sheet.png` 并冻结 ROI、Mask、Prompt 与 SHA256，接受 UI 截断及真实机械边缘/空腔风险。只有独立获得 G1F-1 单次付费调用授权后，才另行实现/审核带显式 live 参数的执行器；限定本 source、固定模型、n=1、最多 1 POST、无重试/fallback，并保留原始输出及严格局部回贴与 mask 内外变化审计。当前代码无法执行真实调用。

本次所有新文件都保存在本独立目录。按用户的本次目录约束，registry、状态与 handoff 采用本地新增记录，未修改仓库或历史文档。
'''
    with (ROOT / 'G1F_0B_FINAL_REPORT.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write(report)
    save_json('handoff_and_registry.json', {'experiment': 'Exp13-G1F-0B', 'status': states,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'source': 'corrosion2',
              'gate': 'PENDING_CHATGPT_GATE', 'G1F_1_authorized': False,
              'canonical_report': str(ROOT / 'G1F_0B_FINAL_REPORT.md'),
              'reason_local_registry': 'All new files confined to this directory per task instruction'})
    excluded = {'input_sha256_manifest.json'}
    save_json('input_sha256_manifest.json', {
        'status': states, 'source': m['source_image'], 'annotation': m['source_json'],
        'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*'))
                  if p.is_file() and p.name not in excluded},
        'api_post_count': 0})
    print(states)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    finalize() if args.finalize else prepare()
