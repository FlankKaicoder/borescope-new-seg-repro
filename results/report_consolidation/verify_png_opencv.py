import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results/report_visualization_retro/figure_manifest.csv"
OUTPUT = ROOT / "results/report_visualization_retro/opencv_decode_audit.json"

with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

decoded = []
failed = []
for row in rows:
    relative_path = row["figure_path"]
    # cv2.imread on Windows may fail on non-ASCII paths; imdecode keeps the
    # decoder check in OpenCV while Path.read_bytes handles Unicode safely.
    encoded = np.frombuffer((ROOT / relative_path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        failed.append(relative_path)
        continue
    decoded.append(
        {
            "figure_path": relative_path,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }
    )

report = {
    "status": "PASS" if not failed and len(decoded) == len(rows) else "FAIL",
    "decoder": "OpenCV cv2.imdecode (Unicode-safe byte input)",
    "opencv_version": cv2.__version__,
    "manifest_rows": len(rows),
    "decoded_png_count": len(decoded),
    "failed": failed,
    "decoded": decoded,
    "no_training": True,
    "no_inference": True,
    "test_accessed": False,
}
OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: report[key] for key in ("status", "opencv_version", "manifest_rows", "decoded_png_count")}, ensure_ascii=False))
raise SystemExit(0 if report["status"] == "PASS" else 1)
