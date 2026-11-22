"""Append-only numerical observations and explicit epoch completeness checks."""
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

from tools.exp17.formal.authorization import HardGate, require


def append_json(path, value):
    line = json.dumps(value, sort_keys=True, allow_nan=False)
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n"); f.flush(); os.fsync(f.fileno())


def quantiles(values):
    values = sorted(float(v) for v in values)
    require(values and all(math.isfinite(v) for v in values), "Missing/nonfinite epoch observations")
    def q(fraction):
        offset = (len(values) - 1) * fraction
        left = int(offset)
        return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (offset - left)
    return {"mean": statistics.fmean(values), "min": values[0], "p25": q(.25),
            "median": q(.5), "p75": q(.75), "max": values[-1]}


def epoch_summary(epoch, rows, *, ema_updates):
    require(len(rows) == 21 and [r["step"] for r in rows] == list(range((epoch-1)*21+1,epoch*21+1)),
            "Epoch has skipped or repeated optimizer updates")
    require(sum(r["images"] for r in rows) == 668 and sum(r["raw_targets"] for r in rows) == 1266,
            "Epoch native source/instance exposure differs")
    require(rows[-1]["images"] == 28 and ema_updates == epoch * 21, "Tail batch/EMA budget differs")
    require(all(r["effective_updates"] == r["step"] and r["optimizer_attempts"] == r["step"]
                and not r["amp_skipped"] for r in rows), "Epoch optimizer or AMP identity differs")
    classes = Counter(); skip = Counter(); groups = {str(c): set() for c in range(7)}
    per_class_losses = {str(c): [] for c in range(7)}
    for r in rows:
        classes.update(r["pair_by_class"]); skip.update(r.get("pair_skips", {}))
        for c, ids in r.get("A_groups_by_class", {}).items(): groups[c].update(ids)
        for c, loss in r.get("pair_loss_by_class", {}).items(): per_class_losses[c].append(loss)
    return {"epoch": epoch, "updates": 21, "effective_updates": epoch*21,
            "images": 668, "raw_instances": 1266, "ema_updates": ema_updates,
            "losses": {name: quantiles([r[name] for r in rows]) for name in
                       ("yolo_loss", "supcon_loss", "pair_loss", "total_loss")},
            "gradient_norm": quantiles([r["gradient_norm"] for r in rows]),
            "clip_coefficient": quantiles([r["clip_coefficient"] for r in rows]),
            "fraction_updates_clipped": sum(r["clip_coefficient"] < 1 for r in rows)/21,
            "pair_by_class": dict(classes), "pair_skips": dict(skip),
            "independent_A_groups_by_class": {c: len(ids) for c, ids in groups.items()},
            "pair_loss_by_class": {c: quantiles(v) for c,v in per_class_losses.items() if v},
            "LR_last": rows[-1]["lr"], "BN_last": rows[-1]["bn_state_sha256"],
            "parameter_state_last": rows[-1].get("parameter_state_sha256"),
            "AMP_skips": 0, "complete": True}


def ensure_space(path, *, required_bytes):
    import shutil
    free = shutil.disk_usage(path).free
    require(free >= required_bytes, f"Insufficient disk space: available={free}, required={required_bytes}")
    return free
