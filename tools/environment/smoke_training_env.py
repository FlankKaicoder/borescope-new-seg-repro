#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result: dict[str, object] = {"imports": {}, "cuda": {}, "ultralytics_model_load": {}, "opencv_decode": {}}
    modules = ["torch", "torchvision", "ultralytics", "cv2", "numpy", "pandas", "sklearn", "shapely", "scipy", "PIL"]
    imported = {}
    for name in modules:
        module = __import__(name)
        imported[name] = module
        result["imports"][name] = getattr(module, "__version__", "installed")  # type: ignore[index]

    torch = imported["torch"]
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    cuda_result = {"available": available, "device_count": count, "torch_cuda": torch.version.cuda, "devices": []}
    if not available or count < 1:
        raise RuntimeError("CUDA smoke failed: no visible CUDA device")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        tensor = torch.arange(16, device=f"cuda:{index}", dtype=torch.float32)
        value = float((tensor * tensor).sum().item())
        cuda_result["devices"].append({"index": index, "name": props.name, "total_memory_bytes": props.total_memory, "tensor_check": value})
    result["cuda"] = cuda_result

    from ultralytics import YOLO
    model = YOLO("yolo11n-seg.yaml")
    result["ultralytics_model_load"] = {"status": "PASS", "source": "yolo11n-seg.yaml", "model_class": type(model.model).__name__}

    cv2 = imported["cv2"]
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV failed to decode {args.image}")
    result["opencv_decode"] = {"status": "PASS", "path": str(args.image), "shape": list(image.shape), "dtype": str(image.dtype)}

    result["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

