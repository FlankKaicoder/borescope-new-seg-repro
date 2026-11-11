"""Validate Exp14 training-label plans without writing a dataset."""

from dataclasses import dataclass

from tools.dataset.build_dataset_v1 import CLASSES, clean_points, polygon_area

PENDING = "PENDING_HUMAN_REVIEW"
REMOVED = "CONFIRMED_REMOVED"
RETAINED = "CONFIRMED_RETAINED"
HOLD = "HOLD"
A_ELIGIBLE = "TRAIN_A_ELIGIBLE"
B_ELIGIBLE = "TRAIN_B_ELIGIBLE"


def polygon_rows(source_id, shapes, size):
    """Reuse the frozen converter's class order, polygon cleanup and precision."""
    if not shapes or min(size) <= 0:
        raise ValueError("Professional annotation and positive source dimensions required")
    width, height = size
    rows = {}
    for index, shape in enumerate(shapes):
        name = shape["label"]
        points, _ = clean_points(shape["points"])
        if (shape["shape_type"] != "polygon" or name not in CLASSES
                or len(points) < 3 or polygon_area(points) <= 0
                or any(not (0 <= x <= width and 0 <= y <= height) for x, y in points)):
            raise ValueError("Invalid professional polygon")
        class_id = CLASSES.index(name)
        coords = " ".join(f"{v:.10f}" for x, y in points for v in (x / width, y / height))
        rows[f"{source_id}_{index:03d}"] = (class_id, f"{class_id} {coords}")
    return rows


@dataclass(frozen=True)
class Review:
    image_sha256: str = ""
    a_status: str = PENDING
    b_status: str = PENDING
    whole_image_verified: bool = False
    source_coordinates_verified: bool = False
    empty_confirmed: bool = False
    region_correspondence_verified: bool = False
    artifact_shortcuts_reviewed: bool = False


def approved_label_plan(rows, decisions, review, *, final_sha256, source_size, final_size, route):
    """Return retained YOLO rows in memory after explicit review of the exact image."""
    if route not in {"A", "B"}:
        raise ValueError("Unknown training route")
    if (review.a_status != A_ELIGIBLE or not review.whole_image_verified
            or not final_sha256 or review.image_sha256 != final_sha256):
        raise ValueError("A approval for this exact image is required")
    if route == "B" and (review.b_status != B_ELIGIBLE
                         or not review.region_correspondence_verified
                         or not review.artifact_shortcuts_reviewed):
        raise ValueError("B requires separate region correspondence and artifact review")
    if not rows or rows.keys() != decisions.keys() or set(decisions.values()) - {REMOVED, RETAINED}:
        raise ValueError("Every original instance must have a confirmed decision")
    if REMOVED not in decisions.values():
        raise ValueError("A pseudo-normal sample requires at least one confirmed removal")
    retained = [line for pid, (_, line) in rows.items() if decisions[pid] == RETAINED]
    if retained and (source_size != final_size or not review.source_coordinates_verified):
        raise ValueError("Cannot reuse source coordinates for retained labels")
    if review.empty_confirmed != (not retained):
        raise ValueError("Empty-target confirmation must agree with retained instances")
    return retained
