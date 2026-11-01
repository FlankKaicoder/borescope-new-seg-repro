"""Offline nine-view diagnostics; no automated visual/geometry/training PASS."""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import ROOT, file_sha, write_json, require


def observe(raw, target=(320, 240)):
    raw = raw.convert('RGB')
    width, height = raw.size
    same_aspect = width * target[1] == height * target[0]
    scale = min(target[0] / width, target[1] / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = raw.resize(size, Image.Resampling.LANCZOS)
    offset = ((target[0] - size[0]) // 2, (target[1] - size[1]) // 2)
    canvas = Image.new('RGB', target, (128, 128, 128))
    canvas.paste(resized, offset)
    valid = Image.new('L', target, 0)
    ImageDraw.Draw(valid).rectangle((offset[0], offset[1], offset[0] + size[0]-1,
                                   offset[1] + size[1]-1), fill=255)
    transform = {'raw_size_wh': [width, height], 'diagnostic_size_wh': list(target),
                 'same_aspect_ratio': same_aspect, 'uniform_scale_requested': scale,
                 'rounded_resize_wh': list(size), 'offset_xy': list(offset),
                 'effective_scale_xy': [size[0] / width, size[1] / height],
                 'resampling': 'LANCZOS', 'raw_cropped': False,
                 'letterbox_padding_pixels': target[0] * target[1] - size[0] * size[1],
                 'information_loss': 'Downsampling discards detail; letterbox has no raw-image data. Rounding may add subpixel anisotropy.',
                 'pixel_alignment_claim': 'NONE; uniform resizing does not establish semantic/pixel correspondence.'}
    return canvas, valid, transform


def delta_stats(original, observed, binary, valid):
    diff = ImageChops.difference(original, observed)
    pixels = list(diff.get_flattened_data())
    target = binary.tobytes()
    allowed = valid.tobytes()
    result = {}
    for label, is_inside in [('inside_mask', True), ('outside_mask', False)]:
        selected = [rgb for rgb, mask, flag in zip(pixels, target, allowed)
                    if flag and (bool(mask) == is_inside)]
        result[label] = {'valid_pixels': len(selected),
            'mean_absolute_rgb_difference': sum(sum(rgb) for rgb in selected) / (3 * len(selected)) if selected else None,
            'max_channel_difference_gt_10_fraction': sum(max(rgb) > 10 for rgb in selected) / len(selected) if selected else None}
    result['note'] = 'Diagnostic comparison only. Excludes letterbox padding; no registration, visual PASS, or Pair usability claim.'
    return diff, result


def audit(raw_path, output):
    require(not output.exists(), 'Refusing audit overwrite')
    with Image.open(raw_path) as raw:
        raw.load()
        raw_format, raw_mode, raw_size = raw.format, raw.mode, list(raw.size)
        raw_rgb = raw.convert('RGB')
    original = Image.open(ROOT / 'inputs/roi.png').convert('RGB')
    alpha = Image.open(ROOT / 'inputs/mask_rgba.png').getchannel('A')
    binary = Image.eval(alpha, lambda value: 255 - value)
    observed, valid, transform = observe(raw_rgb)
    diff, stats = delta_stats(original, observed, binary, valid)
    overlay = original.copy()
    red = Image.new('RGB', original.size, (255, 30, 30))
    overlay.paste(Image.blend(original, red, .4), mask=binary)
    heat = Image.new('RGB', original.size)
    heat.putdata([(min(255, max(rgb)*4), min(255, max(rgb)*2), 0) for rgb in diff.get_flattened_data()])
    windows = {'repair_region': [32, 56, 168, 120], 'mechanical_edge': [24, 96, 184, 132],
               'cavity_observation': [0, 120, 144, 240], 'outside_mask_ui': [0, 0, 320, 60]}

    def pair(box):
        a, b = original.crop(tuple(box)), observed.crop(tuple(box))
        result = Image.new('RGB', (a.width*2+4, a.height), 'white')
        result.paste(a, (0, 0)); result.paste(b, (a.width+4, 0))
        return result

    panels = [('1 Original frozen ROI', original), ('2 Exact polygon overlay', overlay),
              ('3 Raw API output (thumbnail only)', raw_rgb),
              ('4 ROI-sized observation (diagnostic)', observed), ('5 Absolute difference x4', heat),
              ('6 Repair window: original | observation', pair(windows['repair_region'])),
              ('7 Mechanical edge: original | observation', pair(windows['mechanical_edge'])),
              ('8 Cavity window: original | observation', pair(windows['cavity_observation'])),
              ('9 Outside-mask UI: original | observation', pair(windows['outside_mask_ui']))]
    sheet = Image.new('RGB', (1500, 1080), 'white')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(panels):
        x, y = (index % 3)*500, (index//3)*360
        draw.text((x+8, y+8), title, font=font, fill='black')
        view = image.copy()
        if index >= 5:
            # Enlarge local observations for boundary/UI inspection; review-only.
            scale = min(480 / view.width, 310 / view.height)
            view = view.resize((round(view.width*scale), round(view.height*scale)), Image.Resampling.NEAREST)
        else:
            view.thumbnail((480, 310), Image.Resampling.LANCZOS)
        sheet.paste(view, (x+(500-view.width)//2, y+38+(310-view.height)//2))
    output.mkdir(parents=True, exist_ok=False)
    observed.save(output / 'roi_sized_observation.png')
    valid.save(output / 'diagnostic_valid_pixels.png')
    heat.save(output / 'absolute_difference_heatmap.png')
    sheet.save(output / 'nine_view_review_sheet.png')
    write_json(output / 'audit_report.json', {
        'status': 'FIRST_CALL_RAW / PENDING_HUMAN_REVIEW',
        'raw_path': str(raw_path), 'raw_sha256': file_sha(raw_path),
        'raw_format': raw_format, 'raw_mode': raw_mode, 'raw_size_wh': raw_size,
        'geometry_transform': transform, 'diagnostic_change_statistics': stats,
        'observation_windows_xyxy_half_open': windows,
        'window_semantics': 'Fixed visual windows, not authoritative cavity/edge segmentations.',
        'gates': {gate: 'PENDING_HUMAN_REVIEW' for gate in
                  ['CORROSION_REMOVAL', 'MECHANICAL_GEOMETRY', 'OUTSIDE_MASK_FIDELITY', 'PAIR_USABILITY', 'PROVENANCE']},
        'separate_judgments': {'raw_visual_reasonableness': 'PENDING_HUMAN_REVIEW',
                              'raw_pair_geometry': 'PENDING_HUMAN_REVIEW',
                              'postprocessed_pair_usability': 'NOT_ESTABLISHED'},
        'hard_or_feather_generated': False, 'train_pair_eligible': False,
        'batch_generation_authorized': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-output', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    audit(args.raw_output, args.output_directory)
