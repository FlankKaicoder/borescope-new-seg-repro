# Training environment report

Generated UTC: `2026-08-12T11:06:04.504011+00:00`

This is the isolated project training environment. The original Exp00 audit environment remains in `environment_report.md` and `pip_freeze.txt`.

- Virtual environment: `/root/autodl-tmp/borescope-new-seg-repro/.venv`
- Python: `3.12.3`
- Platform: `Linux-5.15.0-94-generic-x86_64-with-glibc2.35`
- GPU visibility: `GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-31992cd1-feb5-9d9c-ab76-93451ae5cb4a)`
- Overall smoke status: **PASS**

`pip check` has one intentional metadata exception: Ultralytics declares `opencv-python`, while this headless server uses the API-compatible `opencv-python-headless` package to avoid GUI dependencies. `cv2` import and real dataset image decoding both passed. `nvidia-ml-py` is installed. No CUDA toolkit or torch package was replaced.

## Imports

| Package/import | Version |
|---|---|
| `torch` | `2.8.0+cu128` |
| `torchvision` | `0.23.0+cu128` |
| `ultralytics` | `8.4.117` |
| `cv2` | `5.0.0` |
| `numpy` | `2.3.2` |
| `pandas` | `3.0.5` |
| `sklearn` | `1.9.0` |
| `shapely` | `2.1.2` |
| `scipy` | `1.18.0` |
| `PIL` | `11.3.0` |

## CUDA smoke

```json
{
  "available": true,
  "device_count": 1,
  "torch_cuda": "12.8",
  "devices": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 2080 Ti",
      "total_memory_bytes": 23069917184,
      "tensor_check": 1240.0
    }
  ]
}
```

## Ultralytics model-load smoke

```json
{
  "status": "PASS",
  "source": "yolo11n-seg.yaml",
  "model_class": "SegmentationModel"
}
```

## OpenCV decode smoke

```json
{
  "status": "PASS",
  "path": "/root/autodl-tmp/损伤训练数据集/1.png",
  "shape": [
    540,
    720,
    3
  ],
  "dtype": "uint8"
}
```
