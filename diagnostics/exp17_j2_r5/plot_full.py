"""Plot published R5 summary CSVs without loading models, images or checkpoints.

python -B plot_full.py --input-dir PUBLIC_SUMMARY --output NEW_FIGURE_DIRECTORY
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BLUE = "#2166ac"
ORANGE = "#d95f02"
GREEN = "#3b7f45"
PURPLE = "#7b4397"
GRAY = "#666666"
J1_MASK = .2727438798549706
SOURCES = ("summary.json", "epochs.csv", "updates.csv", "validation_overall.csv", "primary_J1_R5_per_class.csv",
           "bn_mean_gap.csv", "focus_weight_scales.csv", "gradient_diagnostics.csv")


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(directory, name):
    path = directory / name
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value is None or value == "":
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite observation in public plot input")
    return value


def points(records, field, multiplier=1.):
    values = []
    for row in records:
        value = number(row.get(field))
        epoch = number(row.get("epoch"))
        if value is not None and epoch is not None:
            values.append((epoch, value * multiplier))
    return sorted(values)


def connect(ax, values, *, cadence=1, label=None, color=BLUE, marker="o", linestyle="-", markersize=4):
    """Connect adjacent recorded observations, leaving missing intervals blank."""
    segments = [[]]
    for value in values:
        if segments[-1] and value[0] - segments[-1][-1][0] > cadence:
            segments.append([])
        segments[-1].append(value)
    for index, segment in enumerate(segments):
        if segment:
            x, y = zip(*segment)
            ax.plot(x, y, color=color, label=label if index == 0 else None, marker=marker,
                    markersize=markersize, linestyle=linestyle, linewidth=1.65,
                    markerfacecolor="none" if marker != "o" else color, markeredgewidth=1.1)


def bands(ax, records, field, *, label, color, multiplier=1.):
    grouped = defaultdict(list)
    for row in records:
        epoch, value = number(row.get("epoch")), number(row.get(field))
        if epoch is not None and value is not None:
            grouped[epoch].append(value * multiplier)
    if not grouped:
        return False
    epochs = sorted(grouped)
    means = [statistics.fmean(grouped[epoch]) for epoch in epochs]
    low = [min(grouped[epoch]) for epoch in epochs]
    high = [max(grouped[epoch]) for epoch in epochs]
    # Segment the band too so missing scheduled events cannot acquire values.
    start = 0
    for end in range(1, len(epochs) + 1):
        if end == len(epochs) or epochs[end] - epochs[end - 1] > 5:
            ax.fill_between(epochs[start:end], low[start:end], high[start:end], color=color, alpha=.12, linewidth=0)
            start = end
    connect(ax, list(zip(epochs, means)), cadence=5, label=label, color=color)
    if len(epochs) == 1:
        ax.vlines(epochs[0], low[0], high[0], color=color, linewidth=1.5, alpha=.7)
    return True


def axis(ax, title, ylabel, last_epoch):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="semibold", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    end = max(6, last_epoch + 2)
    ax.set_xlim(0, end)
    interval = 20 if last_epoch >= 60 else 10 if last_epoch >= 30 else 5 if last_epoch >= 10 else 1
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10], nbins=max(3, int(end / interval))))
    ax.grid(axis="y", color="#dedede", linewidth=.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def empty(ax, text):
    ax.text(.5, .5, textwrap.fill(text, 62), transform=ax.transAxes, ha="center", va="center",
            color=GRAY, fontsize=11, linespacing=1.5)


def legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.96, borderpad=.35,
                  fontsize=8.5, **kwargs)


def status_line(summary, epochs):
    last = int(max((number(row.get("epoch")) or 0 for row in epochs), default=0))
    committed = summary.get("journal", {}).get("committed_updates")
    updates = committed if committed is not None else summary.get("counts", {}).get("step_rows")
    text = f"Recorded through Epoch {last}" if last else "No complete epoch record"
    if updates is not None:
        text += f" · {updates:,} committed updates" if committed is not None else f" · {updates:,} recorded update rows"
    status = summary.get("status", "")
    if "SCIENTIFIC_HOLD" in status:
        text += " · scientific HOLD"
    elif summary.get("execution_complete"):
        text += " · full budget completed"
    return text


def save(fig, output, stem, caption):
    fig.text(.035, .025, caption, ha="left", va="bottom", fontsize=8.2, color="#454545", linespacing=1.5)
    fig.savefig(output / (stem + ".png"), dpi=200, facecolor="white")
    fig.savefig(output / (stem + ".pdf"), facecolor="white", metadata={"Title": stem, "Subject": "Observed R5 diagnostic summaries"})
    plt.close(fig)


def performance(directory, output, summary, epoch_rows, last_epoch):
    validation = rows(directory, "validation_overall.csv")
    standard = [row for row in validation if row.get("pass") == "standard"]
    class_rows = [row for row in rows(directory, "primary_J1_R5_per_class.csv") if row.get("metric") == "mask_ap50_95"]
    class_rows.sort(key=lambda row: int(row["class_id"]))
    has_epoch100 = any(number(row.get("epoch")) == 100 and row.get("role") == "master" for row in standard)
    has_last = has_epoch100 and len(class_rows) == 7 and summary.get("primary", {}).get("available", True)
    fig = plt.figure(figsize=(13.2, 9.2 if has_last else 5.6))
    if has_last:
        grid = fig.add_gridspec(2, 2, height_ratios=[1., 1.03], left=.07, right=.97, bottom=.16, top=.86, hspace=.48, wspace=.24)
    else:
        grid = fig.add_gridspec(1, 2, left=.07, right=.97, bottom=.22, top=.77, wspace=.24)
    fig.suptitle("R5 Connected Statistics · Seed42", x=.07, y=.96, ha="left", fontsize=18, fontweight="semibold")
    fig.text(.07, .91 if has_last else .87, status_line(summary, epoch_rows), ha="left", fontsize=10.5, color=GRAY)
    baseline = summary.get("primary", {}).get("J1_seed42_epoch100_Last_MASTER", J1_MASK)
    for position, kind, title in ((grid[0, 0], "M", "Mask AP50–95"), (grid[0, 1], "B", "Box AP50–95")):
        ax = fig.add_subplot(position)
        axis(ax, title, "AP50–95 (%)", last_epoch)
        observed = []
        for role, color, marker, style in (("master", BLUE, "o", "-"), ("ema", ORANGE, "s", "--")):
            values = points([row for row in standard if row.get("role") == role], f"metrics/mAP50-95({kind})", 100.)
            connect(ax, values, cadence=5, label="Scheduled " + role.upper(), color=color, marker=marker, linestyle=style)
            observed.extend(value for _, value in values)
        ax.axhline(0., color=GRAY, linewidth=1., linestyle=":", label="Original J2 scheduled AP = 0")
        if kind == "M":
            ax.axhline(100 * baseline, color=GREEN, linewidth=1.2, linestyle="-.", label="J1 Epoch100 Last MASTER reference")
            observed.append(100 * baseline)
        ax.set_ylim(-.8, max(35., 1.12 * max(observed, default=0)))
        legend(ax, loc="upper left")
        if not any(number(row.get(f"metrics/mAP50-95({kind})")) is not None for row in standard):
            empty(ax, "No scheduled standard VAL result recorded in this input")
    if has_last:
        ax = fig.add_subplot(grid[1, :])
        ax.set_title("Seven-class Mask AP50–95 · Epoch100 Last FP32 MASTER", loc="left", fontsize=11.5, fontweight="semibold", pad=12)
        indexes = list(range(7))
        j1 = [100 * number(row["J1"]) for row in class_rows]
        repaired = [100 * number(row["R5"]) for row in class_rows]
        ax.bar([i - .19 for i in indexes], j1, width=.36, color="#719b79", label="J1 Last MASTER")
        ax.bar([i + .19 for i in indexes], repaired, width=.36, color=BLUE, label="R5 Last MASTER")
        for index, old, new in zip(indexes, j1, repaired):
            ax.text(index, max(old, new) + 1., f"{new-old:+.1f} pp", ha="center", fontsize=9)
        ax.set_xticks(indexes, [textwrap.fill(row["class"], 13) for row in class_rows])
        ax.set_ylim(0, max(10, max(j1 + repaired) * 1.18 + 2))
        ax.set_ylabel("Mask AP50–95 (%)")
        ax.grid(axis="y", color="#dedede", linewidth=.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        legend(ax, loc="upper right", ncols=2)
    missing_last = "\nThe seven-class matched endpoint plot requires an Epoch100 Last MASTER result." if not has_last else ""
    save(fig, output, "01_scheduled_performance",
         "Scheduled standard VAL uses conf=0.001 and normal running statistics. The Epoch100 MASTER is the primary Last endpoint.\n"
         "The J1 line identifies its Epoch100 Last value, independent of R5 epoch. Best selections remain supplementary. Recorded points are shown without imputed events." + missing_last)


def training(directory, output, summary, epoch_rows, last_epoch):
    fig = plt.figure(figsize=(13.2, 8.7))
    grid = fig.add_gridspec(2, 3, left=.075, right=.97, bottom=.16, top=.85, hspace=.48, wspace=.35)
    fig.suptitle("Training losses and clipping", x=.075, y=.96, ha="left", fontsize=18, fontweight="semibold")
    fig.text(.075, .91, status_line(summary, epoch_rows), ha="left", fontsize=10.5, color=GRAY)
    specifications = (("yolo_loss_mean", 1., "Native YOLO loss", BLUE),
                      ("supcon_loss_mean", 8., "SupCon loss ×8", GREEN),
                      ("pair_loss_mean", 4., "Pair ranking loss ×4", PURPLE))
    for column, (field, multiplier, title, color) in enumerate(specifications):
        ax = fig.add_subplot(grid[0, column])
        axis(ax, title, "Weighted loss" if multiplier != 1 else "Native loss", last_epoch)
        values = points(epoch_rows, field, multiplier)
        connect(ax, values, color=color)
        if not values:
            empty(ax, "No complete epoch loss record")
    ax = fig.add_subplot(grid[1, :2])
    axis(ax, "Unscaled parameter gradients before clipping", "Gradient L2 (log scale)", last_epoch)
    all_values = []
    for field, color, label, marker in (("gradient_norm_mean", BLUE, "Epoch mean", "o"),
                                       ("gradient_norm_median", ORANGE, "Epoch median", "s")):
        values = points(epoch_rows, field)
        connect(ax, values, label=label, color=color, marker=marker)
        all_values.extend(value for _, value in values)
    ax.axhline(10., color=GRAY, linewidth=1., linestyle=":", label="Clip10 threshold")
    if all(value > 0 for value in all_values):
        ax.set_yscale("log")
        ax.set_ylim(5., max(20., max(all_values, default=10.) * 1.3))
    legend(ax, loc="upper right")
    if not all_values:
        empty(ax, "No complete epoch gradient record")
    ax = fig.add_subplot(grid[1, 2])
    axis(ax, "Updates triggering Clip10", "Updates clipped (%)", last_epoch)
    values = points(epoch_rows, "fraction_updates_clipped", 100.)
    connect(ax, values, color=BLUE)
    ax.set_ylim(-3, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    if not values:
        empty(ax, "No complete epoch clipping record")
    skips = summary.get("counts", {}).get("AMP_skips")
    skip_text = f" Recorded AMP skipped updates {skips}." if skips is not None else ""
    save(fig, output, "02_losses_and_clipping",
         "Each loss point is the recorded arithmetic mean over the epoch's optimizer batches. SupCon and Pair weights are applied once.\n"
         "Gradient norms are measured after AMP unscale and before global clipping. The clipping fraction does not measure AdamW update scaling." + skip_text)


def mechanisms(directory, output, summary, epoch_rows, last_epoch):
    gaps = rows(directory, "bn_mean_gap.csv")
    gradients = rows(directory, "gradient_diagnostics.csv")
    weights = rows(directory, "focus_weight_scales.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    fig.subplots_adjust(left=.075, right=.97, bottom=.17, top=.85, hspace=.45, wspace=.26)
    fig.suptitle("Recorded TRAIN diagnostics", x=.075, y=.96, ha="left", fontsize=18, fontweight="semibold")
    fig.text(.075, .91, status_line(summary, epoch_rows), ha="left", fontsize=10.5, color=GRAY)
    for ax, title, entries in ((axes[0, 0], "Backbone BN input mean gaps", [("model.0.bn", "Backbone0", BLUE), ("model.4.cv1.bn", "Backbone4 cv1", ORANGE)]),
                               (axes[0, 1], "Neck16 BN input mean gaps", [("model.16.cv1.bn", "Neck16 cv1", BLUE), ("model.16.cv2.bn", "Neck16 cv2", ORANGE)])):
        axis(ax, title, "Normalized mean gap D", last_epoch)
        present = False
        for layer, label, color in entries:
            present |= bands(ax, [row for row in gaps if row.get("layer") == layer], "D", label=label, color=color)
        ax.axhline(3., color=GRAY, linewidth=1., linestyle=":", label="Early-gate bound D=3")
        values = [number(row.get("D")) for row in gaps if row.get("layer") in {value[0] for value in entries}]
        ax.set_ylim(0, max(3.3, 1.12 * max((v for v in values if v is not None), default=0)))
        legend(ax, loc="upper left")
        if not present:
            empty(ax, "No sentinel measurement recorded")
    ax = axes[1, 0]
    axis(ax, "Pair4 relative to YOLO + SupCon8 gradients", "Pair4 / main gradient L2 (%)", last_epoch)
    present = False
    for domain, label, color in (("all_trainable", "All trainable", BLUE), ("backbone", "Backbone", ORANGE),
                                  ("model.13", "Neck13", GREEN), ("model.16", "Neck16", PURPLE)):
        present |= bands(ax, [row for row in gradients if row.get("domain") == domain], "pair4_to_main_ratio",
                         label=label, color=color, multiplier=100.)
    ax.set_ylim(bottom=0)
    legend(ax, loc="upper left", ncols=2)
    if not present:
        empty(ax, "No fixed-probe component gradient measurement")
    ax = axes[1, 1]
    axis(ax, "Neck16 convolution-weight scale", "FP32 weight L2", last_epoch)
    present = False
    for parameter, label, color in (("model.16.cv1.conv.weight", "Neck16 cv1", BLUE), ("model.16.cv2.conv.weight", "Neck16 cv2", ORANGE)):
        present |= bands(ax, [row for row in weights if row.get("parameter") == parameter], "L2", label=label, color=color)
    ax.set_ylim(bottom=0)
    legend(ax, loc="upper left")
    if not present:
        empty(ax, "No saved weight-scale measurement")
    save(fig, output, "03_bn_pair_and_weight_scale",
         "Lines show means and bands show the observed min–max across recorded fixed TRAIN probes. D is each probe's median normalized channel mean gap.\n"
         "BN gaps compare TRAIN A inputs with saved pre-probe running buffers. D=3 is the predeclared early-gate bound. Pair ratios already include weight4.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new or empty output directory to preserve prior figures")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.titlelocation": "left"})
    path = args.input_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    epoch_rows = rows(args.input_dir, "epochs.csv")
    last_epoch = max((number(row.get("epoch")) or 0 for row in epoch_rows), default=0)
    performance(args.input_dir, args.output, summary, epoch_rows, last_epoch)
    training(args.input_dir, args.output, summary, epoch_rows, last_epoch)
    mechanisms(args.input_dir, args.output, summary, epoch_rows, last_epoch)
    manifest = {"schema": "EXP17_R5_OFFLINE_FIGURE_MANIFEST_V1", "source_status": summary.get("status"),
        "plotter_sha256": file_sha(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "source_files_sha256": {name: file_sha(args.input_dir / name) for name in SOURCES if (args.input_dir / name).is_file()},
        "figures": [{"file": path.name, "sha256": file_sha(path), "bytes": path.stat().st_size} for path in sorted(args.output.iterdir())],
        "reference_values": {"J1_seed42_epoch100_Last_MASTER_mask_AP50_95": summary.get("primary", {}).get("J1_seed42_epoch100_Last_MASTER", J1_MASK),
                             "original_J2_scheduled_standard_AP": 0.},
        "range_definition": "Observed min-max over recorded fixed TRAIN probes, with no confidence interval",
        "new_forward": 0, "new_backward": 0, "new_VAL": 0, "TEST_access": 0, "imputed_observations": 0}
    with (args.output / "plot_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PLOT_COMPLETE", "figures": 3, "formats": ["PNG", "PDF"], "recorded_last_epoch": last_epoch}))


if __name__ == "__main__":
    main()
