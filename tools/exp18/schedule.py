"""H100/H200 schedule mathematics; never constructs a real trainer."""
from __future__ import annotations

import hashlib
from decimal import Decimal, localcontext
from .common import require


def epoch_seed(seed, epoch, epochs=200):
    require(type(seed) is int and seed >= 0 and epochs in (100, 200), "Invalid epoch seed profile")
    require(type(epoch) is int and 0 <= epoch < epochs, "Epoch outside profile")
    return int.from_bytes(hashlib.sha256(f"exp14-a11-full:v1:{seed}:{epoch}".encode("ascii")).digest()[:8], "little")


def point(update, epochs=200):
    require(type(update) is int and epochs in (100, 200) and 1 <= update <= 21 * epochs, "Update outside profile")
    epoch0, batch0 = divmod(update - 1, 21)
    base = .01 * (max(1 - epoch0 / epochs, 0) * (1 - .01) + .01)
    if update - 1 < 63:
        # Match native np.interp operation order, independently checked below.
        import numpy as np
        weight = float(np.interp(update - 1, [0, 63], [0., base]))
        bias = float(np.interp(update - 1, [0, 63], [.1, base]))
    else:
        weight = bias = base
    return {"step": update, "epoch": epoch0 + 1, "batch_index": batch0,
            "images": 32 if batch0 < 20 else 28, "lr": [weight, weight, bias],
            "betas": [.937, .999], "weight_decay": [.0005, 0., 0.], "accumulate": 1}


def mathematical_lr(update, epochs=200):
    require(epochs in (100, 200) and 1 <= update <= 21 * epochs, "Update outside independent mathematical profile")
    with localcontext() as c:
        c.prec = 60
        base = Decimal(".01") * (1 - Decimal(".99") * Decimal((update - 1) // 21) / Decimal(epochs))
        t = Decimal(update - 1) / 63
        weight, bias = (base * t, Decimal(".1") * (1 - t) + base * t) if update <= 63 else (base, base)
        return [float(weight), float(weight), float(bias)]
