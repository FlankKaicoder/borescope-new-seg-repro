"""Independent, CPU-only alignment validation for frozen G3-C0/C2 assets."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def features(image, kind):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if kind == "SIFT":
        detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.025)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
        norm = cv2.NORM_HAMMING
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints or [], descriptors, norm


def correspondences(source, output, kind):
    ka, da, norm = features(source, kind)
    kb, db, _ = features(output, kind)
    if da is None or db is None or len(db) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=2)
    ranked = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < (0.75 if kind == "SIFT" else 0.8) * pair[1].distance:
            m = pair[0]
            ranked.append((m.distance / max(pair[1].distance, 1e-9), m))
    used_source, used_output, a, b = set(), set(), [], []
    for _, match in sorted(ranked, key=lambda item: item[0]):
        pa = np.asarray(ka[match.queryIdx].pt)
        pb = np.asarray(kb[match.trainIdx].pt)
        # A 24-pixel source grid prevents dense repeated texture from dominating.
        ca = tuple(np.floor(pa / 24).astype(int))
        cb = tuple(np.floor(pb / 24).astype(int))
        if ca in used_source or cb in used_output:
            continue
        used_source.add(ca)
        used_output.add(cb)
        a.append(pa)
        b.append(pb)
    return np.asarray(a, np.float64).reshape(-1, 2), np.asarray(b, np.float64).reshape(-1, 2)


def estimate(output_points, source_points, model):
    if len(source_points) < (3 if model == "AFFINE" else 2):
        return None
    cv2.setRNGSeed(20261109)
    if model == "AFFINE":
        matrix, _ = cv2.estimateAffine2D(output_points, source_points, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0, maxIters=4000,
                                         confidence=0.995, refineIters=10)
    else:
        matrix, _ = cv2.estimateAffinePartial2D(output_points, source_points, method=cv2.RANSAC,
                                               ransacReprojThreshold=3.0, maxIters=4000,
                                               confidence=0.995, refineIters=10)
    return matrix if matrix is not None and np.isfinite(matrix).all() else None


def predict(matrix, points):
    return np.asarray(points) @ matrix[:, :2].T + matrix[:, 2]


def holdout_evaluation(output_points, source_points, model, source_shape):
    # Spatially interleaved grid cells are held out before fitting.
    cell = np.floor(source_points / 64).astype(int)
    held = ((cell[:, 0] * 3 + cell[:, 1] * 5) % 5) == 0
    if held.sum() < 3 or (~held).sum() < 4:
        return {"holdout_count": int(held.sum()), "holdout_median_px": None,
                "holdout_p90_px": None, "status": "INSUFFICIENT_SPATIAL_HOLDOUT"}
    matrix = estimate(output_points[~held], source_points[~held], model)
    if matrix is None:
        return {"holdout_count": int(held.sum()), "holdout_median_px": None,
                "holdout_p90_px": None, "status": "FIT_FAILED"}
    error = np.linalg.norm(predict(matrix, output_points[held]) - source_points[held], axis=1)
    threshold = max(2.0, 0.005 * math.hypot(source_shape[1], source_shape[0]))
    return {"holdout_count": int(held.sum()), "holdout_median_px": float(np.median(error)),
            "holdout_p90_px": float(np.percentile(error, 90)),
            "status": "HOLDOUT_PASS" if np.median(error) <= threshold and np.percentile(error, 90) <= 2 * threshold
            else "HOLDOUT_FAIL"}


def target_neighborhood_coverage(matrix, output_shape, target_mask):
    h, w = target_mask.shape
    ring = cv2.dilate(target_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(target_mask))
    if cv2.countNonZero(ring) == 0:
        return None
    oh, ow = output_shape[:2]
    valid = cv2.warpAffine(np.full((oh, ow), 255, np.uint8), matrix, (w, h),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return float(cv2.countNonZero(cv2.bitwise_and(valid, ring)) / cv2.countNonZero(ring))


def compare_pair(source_path, output_path, existing, target_mask_path=None):
    source, output = read_rgb(source_path), read_rgb(output_path)
    record = {"source_id": source_path.parent.name, "variant": output_path.parent.name,
              "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
              "prior_geometry": existing}
    mask = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE) if target_mask_path else None
    methods = {}
    for detector in ("SIFT", "ORB"):
        a, b = correspondences(source, output, detector)
        methods[detector] = {"matches": int(len(a)), "models": {}}
        for model in ("SIMILARITY", "AFFINE"):
            matrix = estimate(b, a, model)
            holdout = holdout_evaluation(b, a, model, source.shape) if len(a) else {
                "holdout_count": 0, "holdout_median_px": None, "holdout_p90_px": None,
                "status": "NO_MATCHES"}
            values = {"holdout": holdout}
            if matrix is not None:
                values["matrix_output_to_source"] = matrix.tolist()
                if mask is not None and mask.shape == source.shape[:2]:
                    values["defect_neighborhood_coverage"] = target_neighborhood_coverage(
                        matrix, output.shape, mask)
            methods[detector]["models"][model] = values
    record["methods"] = methods
    return record


def synthetic_known_transform():
    rng = np.random.default_rng(20261109)
    image = np.zeros((360, 480), np.uint8)
    for _ in range(150):
        x, y = rng.integers(12, 468), rng.integers(12, 348)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 8)),
                   int(rng.integers(80, 255)), -1)
    cv2.putText(image, "STRUCTURE 42", (40, 180), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, 230, 2, cv2.LINE_AA)
    truth = np.asarray([[0.83, -0.04, 28.0], [0.04, 0.83, 19.0]], np.float64)
    warped = cv2.warpAffine(image, cv2.invertAffineTransform(truth), (480, 360))
    kp_a, des_a = cv2.SIFT_create(nfeatures=2000).detectAndCompute(image, None)
    kp_b, des_b = cv2.SIFT_create(nfeatures=2000).detectAndCompute(warped, None)
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_b, des_a, k=2)
    pa, pb = [], []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            pa.append(kp_b[pair[0].queryIdx].pt)
            pb.append(kp_a[pair[0].trainIdx].pt)
    matrix = estimate(np.asarray(pa), np.asarray(pb), "AFFINE")
    corners = np.asarray([[30, 30], [450, 30], [450, 330], [30, 330]], np.float64)
    error = float(np.median(np.linalg.norm(predict(matrix, corners) - predict(truth, corners), axis=1))) if matrix is not None else None
    return {"known_transform_corner_median_error_px": error, "match_count": len(pa),
            "pass": error is not None and error <= 2.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0-root", type=Path, required=True)
    parser.add_argument("--c2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c0 = {r["source_id"]: r for r in csv.DictReader(
        (args.c0_root / "geometry_review.csv").open(encoding="utf-8-sig", newline=""))}
    c2 = {r["source_id"]: r for r in csv.DictReader(
        (args.c2_root / "geometry_review.csv").open(encoding="utf-8-sig", newline=""))}
    rows = []
    for sid, row in sorted(c0.items(), key=lambda item: int(item[0])):
        folder = args.c0_root / sid
        mask = folder / "valid_target_mask.png"
        rows.append(compare_pair(folder / "original.png", folder / "raw_output.png",
                                 row["geometry_status"], mask if mask.exists() else None))
    for sid, row in sorted(c2.items(), key=lambda item: int(item[0])):
        folder = args.c2_root / sid
        mask = folder / "valid_target_mask.png"
        if not mask.exists() and (args.c0_root / sid / "valid_target_mask.png").exists():
            mask = args.c0_root / sid / "valid_target_mask.png"
        rows.append(compare_pair(folder / "original.png", folder / "raw_output.png",
                                 row["v2_geometry_status"], mask if mask.exists() else None))
    result = {"protocol": "G3-D1 independent spatial holdout v1", "pair_count": len(rows),
              "synthetic_control": synthetic_known_transform(), "pairs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"pair_count": len(rows), "synthetic_control": result["synthetic_control"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
