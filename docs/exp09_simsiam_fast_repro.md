# Exp09 SimSiam fast reproduction

Status: **STOP / BACKBONE_TRANSFER_HARD_GATE**. `test_accessed=false`.

SSL used only the 668 frozen TRAIN images; VAL and TEST images seen were both zero. The encoder was official YOLO11n-seg layers 0-10 only, excluding neck and head (1365472 parameters, 240 state tensors). Frozen settings were batch 32, image size 512, 100 epochs, SGD momentum 0.9, weight decay 1e-4, learning rate 0.00625, and seed 42.

All 100 SSL epochs completed. Loss and checkpoint values were finite. Minimum feature std was 0.105358; minimum projector embedding std was 1.007856; no collapse was detected. Adapted-backbone SHA256: `31d918c3d290d91ed38befc1fa1902443d99940e37e5e9de2945ce5e94ae1a24`.

Transfer key audit reported expected=240, missing=0, unexpected=0, and 120 tensors changed relative to COCO. After saving and reloading the downstream YOLO checkpoint, however, only 160/240 backbone tensors were byte-for-byte equal to the SimSiam export. Example tensor `0.bn.running_mean` had COCO/SimSiam/reloaded hashes `f1e972f52aab9a528d9439e30b907b9563ef9d33bb2e15563952f7caedf0a50a` / `d762bdb605dbf5f3b6bb00e081555ff60745dc5a780c53db709d8f74fab54c0b` / `81c406e5aae32d91dff8d0a319cd72373df06fd5df592da21911c0aea971a426`. Complete backbone migration therefore was not proven and task Hard Gate 10 fired.

Per the gate, downstream 100-epoch training and VAL evaluation were not run. No classification of SimSiam as positive/no-clear-gain/negative is valid. No repair search was performed, and TEST was never accessed.
