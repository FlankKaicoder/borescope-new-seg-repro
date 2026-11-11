"""Bind Exp14-A2 to the reviewed cohort and inspect existing Aligned support."""

import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps

from tools.exp14.audit_admission import read_json, read_rows, require, sha256_file, under, verify_train
from tools.exp14.import_human_review import (
    A0_DIR, FLAGS, GATE, INPUTS, expected_priority, final_image, read_page_bundle,
    source_annotations, unique_index, validate_schema, verified_file, verify_bundle,
    verify_record_identity,
)


A_FIELDS = {
    "variant_id", "source_id", "group_id", "version", "split", "repair_class", "final_image_choice",
    "final_image_path", "final_image_sha256", "final_image_size_wh", "source_image_path",
    "source_image_sha256", "annotation_path", "annotation_sha256", "delete_polygon_ids",
    "keep_polygon_ids", "label_plan", "review_sha256", "bundle_id", "release_status",
}
B_FIELDS = {"region_polygon_ids", "region_coordinate_grid", "valid_support_mask_path", "valid_support_mask_sha256"}
A1_INPUTS = (
    "review_import_audit.json", "exp14_a1_gate_summary.json", "review_import_details.csv",
    "label_decision_summary.csv", "train_a_candidate_manifest.csv", "train_b_pair_candidate_manifest.csv",
)
READY = "CORE_SOURCE_SCREEN_PASS"
HOLD = "TRAINING_HOLD"


def cohort_rows(a_rows, b_rows, details, expected=(22, 21)):
    a = unique_index(a_rows, "variant_id")
    b = unique_index(b_rows, "variant_id")
    require((len(a), len(b)) == expected and b.keys() < a.keys(), "A1 cohort counts or intersection differ")
    require({k for k, d in details.items() if d["a_status"] == "VERIFIED_CANDIDATE"} == a.keys()
            and {k for k, d in details.items() if d["b_status"] == "VERIFIED_CANDIDATE"} == b.keys(),
            "A1 details and candidate sets differ")
    for key, row in b.items():
        require(set(a[key]) == A_FIELDS and set(row) == A_FIELDS | B_FIELDS
                and all(row[k] == a[key][k] for k in A_FIELDS), "A/B candidate identity differs")
    for name in ("source_id", "group_id"):
        unique_index(a_rows, name)
        unique_index(b_rows, name)
    return a, b


def match_candidate(row, a0, imported, human, entry, audit, labels):
    variant = row["variant_id"]
    binding = {
        "variant_id": variant, "source_id": a0["source_id"], "group_id": a0["group_id"],
        "version": a0["version"], "split": "train", "repair_class": a0["repair_class"],
        "final_image_choice": "aligned_output", "final_image_path": a0["aligned_output_path"],
        "final_image_sha256": a0["aligned_output_sha256"], "final_image_size_wh": a0["source_size_wh"],
        "source_image_path": a0["source_path"], "source_image_sha256": a0["source_sha256"],
        "annotation_path": a0["annotation_path"], "annotation_sha256": a0["annotation_sha256"],
        "review_sha256": audit["review_sha256"], "bundle_id": audit["bundle_id"],
        "release_status": "PENDING_CHATGPT_DATA_GATE", "label_plan": "EMPTY_TARGET_CONFIRMED",
        "region_coordinate_grid": "ORIGINAL_SOURCE_PIXELS", "valid_support_mask_path": a0["valid_mask_path"],
        "valid_support_mask_sha256": a0["valid_mask_sha256"],
    }
    require(all(row[k] == value for k, value in binding.items()), f"{variant}: A1 candidate binding differs")
    require(imported["a_human_decision"] == imported["b_human_decision"] == human["a_decision"]
            == human["b_decision"] == "PASS" and imported["image_verified"] == "True"
            and all(imported[k] == binding[k] for k in (
                "variant_id", "source_id", "group_id", "repair_class", "final_image_choice",
                "final_image_sha256", "review_sha256", "bundle_id"))
            and all(human[k] for k in FLAGS) and json.loads(imported["human_flags"]) == {k: human[k] for k in FLAGS},
            f"{variant}: human A/B evidence differs")
    deleted = json.loads(row["delete_polygon_ids"])
    require(len(deleted) == len(set(deleted)) == len(labels) and set(deleted) == labels.keys()
            and json.loads(row["keep_polygon_ids"]) == [] and json.loads(row["region_polygon_ids"]) == deleted
            and set(a0["selected_polygon_ids"].split("|")) == labels.keys()
            and all(x["decision"] == "DELETE" for x in human["polygons"])
            and json.loads(imported["label_decisions"]) == dict.fromkeys(deleted, "DELETE"),
            f"{variant}: DELETE, region or empty-target evidence differs")
    require(entry["images"]["aligned_output"]["sha256"] == row["final_image_sha256"],
            f"{variant}: Aligned is missing from the reviewed bundle")
    return deleted


def load_cohort(root, a1):
    a0_root = under(root, A0_DIR)
    inputs = {name: sha256_file(under(a0_root, name)) for name in INPUTS}
    a1_hashes = {name: sha256_file(under(a1, name)) for name in A1_INPUTS}
    audit = read_json(under(a1, "review_import_audit.json"))
    summary = read_json(under(a1, "exp14_a1_gate_summary.json"))
    require(summary == audit["audit_summary"] and summary["gate"] == GATE
            and summary["training_data_released"] is False and summary["historical_trusted_pair_pass"] == 0
            and audit["input_sha256"] == inputs, "A1 audit or A0 input hashes differ")
    review_path = verified_file(a0_root, audit["review_file"], audit["review_sha256"])
    page_path = verified_file(a0_root, audit["html_bundle_file"], audit["html_sha256"])
    export = validate_schema(review_path.read_bytes())
    require(export["bundle_id"] == audit["bundle_id"], "A1 human-review bundle differs")
    manifest = unique_index(read_rows(under(a0_root, "candidate_manifest.csv")), "variant_id")
    a0_reviews = unique_index(read_rows(under(a0_root, "candidate_review.csv")), "variant_id")
    entries = verify_bundle(export, read_page_bundle(page_path), inputs, expected_priority(manifest, a0_reviews))
    plan = read_rows(under(a0_root, "label_change_plan.csv"))
    human = unique_index(export["records"], "variant_id")
    imported = unique_index(read_rows(under(a1, "review_import_details.csv")), "variant_id")
    require(imported.keys() == manifest.keys(), "A1 variant coverage differs")
    a, b = cohort_rows(read_rows(under(a1, "train_a_candidate_manifest.csv")),
                       read_rows(under(a1, "train_b_pair_candidate_manifest.csv")), imported)
    require(len(human) == 24 and summary["A"]["independent_sources"] == len(a)
            and summary["B"]["independent_sources"] == len(b) and summary["b_removed_region_candidates"] == 24,
            "A1 source or region counts differ")
    train_root, train, train_report = verify_train(root)
    result = []
    imported_labels = read_rows(under(a1, "label_decision_summary.csv"))
    for key, row in b.items():
        prior = manifest[key]
        labels, shapes, size = source_annotations(root, prior, train[row["source_id"]])
        frozen_plan = unique_index([p for p in plan if p["variant_id"] == key], "polygon_id")
        verify_record_identity(prior, human[key], entries[key], labels, frozen_plan)
        removed = match_candidate(row, prior, imported[key], human[key], entries[key], audit, labels)
        reviewed_labels = unique_index([p for p in imported_labels if p["variant_id"] == key], "polygon_id")
        require(reviewed_labels.keys() == labels.keys() and all(
            p["decision"] == "DELETE" and p["source_id"] == row["source_id"] and p["group_id"] == row["group_id"]
            and p["final_image_sha256"] == row["final_image_sha256"]
            and p["class_id"] == str(labels[pid][0]) for pid, p in reviewed_labels.items()),
            f"{key}: A1 per-instance DELETE record differs")
        image = final_image(root, prior, human[key], entries[key])
        support = verified_file(root, row["valid_support_mask_path"], row["valid_support_mask_sha256"])
        diagnostic = verified_file(root, prior["geometry_diagnostic_path"], prior["geometry_diagnostic_sha256"])
        result.append({"candidate": row, "image": image, "support": support, "diagnostic": diagnostic,
                       "region_ids": removed, "labels": labels, "shapes": shapes, "size": size})
    require(sum(len(r["region_ids"]) for r in result) == 24, "Core Polygon count differs")
    require(all(sha256_file(under(a1, name)) == digest for name, digest in a1_hashes.items())
            and all(sha256_file(under(a0_root, name)) == digest for name, digest in inputs.items())
            and sha256_file(review_path) == audit["review_sha256"] and sha256_file(page_path) == audit["html_sha256"],
            "Audit inputs changed during verification")
    excluded = [{"variant_id": key, "a1_a_status": d["a_status"], "a1_b_status": d["b_status"],
                 "reason": "A1_A_ONLY_OUTSIDE_COMMON_COHORT" if key in a else
                           "OUTSIDE_PRIORITY_REVIEW" if d["review_scope"] == "OUTSIDE_PRIORITY" else
                           "A1_NOT_VERIFIED_FOR_BOTH_ROUTES"}
                for key, d in imported.items() if key not in b]
    provenance = {"a1_inputs_sha256": a1_hashes, "a0_inputs_sha256": inputs,
                  "human_review_sha256": audit["review_sha256"], "bundle_id": audit["bundle_id"],
                  "review_html_sha256": audit["html_sha256"]}
    return result, train_root, train, train_report, excluded, provenance


def image_mode(path, mode):
    with Image.open(path) as image:
        return image.convert(mode)


def polygon_mask(size, shape):
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in shape["points"]], fill=255)
    return mask


def pixel_count(binary):
    return binary.histogram()[255]


def support_metrics(original, aligned, valid, regions):
    """Compute a screen of conservative validity, exact black and visible source loss."""
    require(original.size == aligned.size == valid.size and not any(valid.histogram()[1:255]),
            "Full-frame support mask size or encoding differs")
    invalid = ImageOps.invert(valid)
    red, green, blue = aligned.split()
    max_rgb = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    exact_black = max_rgb.point([255] + [0] * 255)
    fill = ImageChops.multiply(invalid, exact_black)
    original_dark = original.convert("L").point([255] * 25 + [0] * 231)
    darker_border = ImageChops.multiply(fill, original_dark)
    lost_visible = ImageChops.multiply(fill, ImageOps.invert(original_dark))
    w, h = valid.size
    metrics = {
        "frame_pixels": w * h, "valid_pixels": pixel_count(valid),
        "valid_fraction": pixel_count(valid) / (w * h), "invalid_pixels": pixel_count(invalid),
        "invalid_bbox_xyxy": invalid.getbbox(), "exact_black_outside_support_pixels": pixel_count(fill),
        "fill_over_dark_source_pixels_luma_lt25": pixel_count(darker_border),
        "fill_over_visible_source_pixels_luma_ge25": pixel_count(lost_visible),
        "visible_loss_bbox_xyxy": lost_visible.getbbox(),
        "invalid_tiles_3x3": [[pixel_count(invalid.crop((x*w//3, y*h//3, (x+1)*w//3, (y+1)*h//3)))
                              for x in range(3)] for y in range(3)],
        "regions": {},
    }
    for pid, shape in regions.items():
        target = polygon_mask(valid.size, shape)
        count = pixel_count(target)
        require(count > 0, "Empty original Polygon raster")
        covered = pixel_count(ImageChops.multiply(target, valid))
        metrics["regions"][pid] = {"target_pixels": count, "valid_pixels": covered,
                                   "valid_fraction": covered / count}
    return metrics, invalid, lost_visible


def screen_core(root, items, visual_dir=None):
    reports = []
    strips = []
    for item in items:
        row = item["candidate"]
        source = verified_file(root, row["source_image_path"], row["source_image_sha256"])
        original = image_mode(source, "RGB")
        aligned = image_mode(item["image"].path, "RGB")
        valid = image_mode(item["support"], "L")
        metrics, invalid, loss = support_metrics(
            original, aligned, valid, {pid: item["shapes"][pid] for pid in item["region_ids"]})
        reports.append({"variant_id": row["variant_id"], "aligned_sha256": item["image"].sha256,
                        "source_sha256": row["source_image_sha256"], "valid_mask_sha256": row["valid_support_mask_sha256"],
                        "repair_class": row["repair_class"], **metrics})
        if visual_dir is not None:
            strips.append(render_strip(row["variant_id"], original, aligned, invalid, loss, metrics, visual_dir))
    if visual_dir is not None:
        render_sheets(strips, visual_dir)
    return reports


def render_strip(variant, original, aligned, invalid, loss, metrics, directory):
    red = Image.new("RGB", original.size, "#ec2358")
    amber = Image.new("RGB", original.size, "#ffbd38")
    overlay = Image.composite(Image.blend(original, red, 0.55), original, invalid)
    overlay = Image.composite(Image.blend(original, amber, 0.7), overlay, loss)
    strip = Image.new("RGB", (1280, 320), "#152130")
    draw = ImageDraw.Draw(strip)
    draw.text((9, 6), f'{variant}  |  valid={metrics["valid_fraction"]:.2%}  |  '
                      f'visible source replaced by black={metrics["fill_over_visible_source_pixels_luma_ge25"]:,} px',
              fill="white")
    for i, (im, label) in enumerate(((original, "Original"), (aligned, "Aligned"), (overlay, "Invalid: red | visible source replaced by black: amber"))):
        x = i * 425 + 5
        draw.text((x + 4, 27), label, fill="white")
        panel = ImageOps.contain(im, (414, 265))
        strip.paste(panel, (x + (414-panel.width)//2, 48 + (265-panel.height)//2))
    path = directory / f"{variant}_support.png"
    with path.open("xb") as file:
        strip.save(file, format="PNG")
    return strip


def render_sheets(strips, directory):
    for offset in range(0, len(strips), 3):
        rows = strips[offset:offset+3]
        sheet = Image.new("RGB", (1280, 320 * len(rows)))
        for i, strip in enumerate(rows):
            sheet.paste(strip, (0, 320*i))
        with (directory / f"contact_{offset//3+1:02}.png").open("xb") as file:
            sheet.save(file, format="PNG")


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")



def main():
    import argparse
    import tempfile
    from datetime import date

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--a1", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root.resolve(strict=True)
    a1 = args.a1.resolve(strict=True)
    require(a1.is_dir() and a1.is_relative_to(root) and a1 != root, "A1 must be inside the private root")
    items, _, _, _, _, provenance = load_cohort(root, a1)
    output = Path(tempfile.mkdtemp(prefix=f"exp14_a2_core_freeze_{date.today():%Y%m%d}_", dir=root))
    evidence = output / "support_visuals"
    evidence.mkdir()
    write_json(output / "aligned_support_analysis.json",
               {"provenance": provenance, "reports": screen_core(root, items, evidence)})
    print(output)


if __name__ == "__main__":
    main()
