"""Publication-readable plots from the existing five-epoch CSVs only."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path):
    with Path(path).open(newline="", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--gaps", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 180})
    data = rows(args.comparison)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for name, label, color in (("J1", "J1 Seed42", "#28689b"), ("Original_J2", "Original J2", "#9b5b38"), ("R5", "Connected R5", "#278663")):
        selected = [r for r in data if r["method"] == name]
        x = [int(r["epoch"]) for r in selected]
        for axis, field in zip(axes, ("yolo_mean", "pair4_mean", "preclip_norm_mean")):
            axis.plot(x, [float(r[field]) for r in selected], "o-", color=color, label=label, linewidth=1.8, markersize=4)
    for axis, title, label in zip(axes, ("Native main-task loss", "Weighted Pair loss", "Joint gradient before clipping"),
                                ("Mean YOLO loss", "Mean Pair4 loss", "Mean L2 norm")):
        axis.set(title=title, xlabel="Completed epoch", ylabel=label, xticks=range(1, 6))
        axis.grid(alpha=.2)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[2].set_yscale("log")
    axes[2].axhline(10, color="#555555", linestyle="--", linewidth=1, label="Clip10")
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.savefig(args.output / "early_training.png")
    plt.close(fig)

    gaps = rows(args.gaps)
    layers = ["model.0.bn", "model.4.cv1.bn", "model.16.cv1.bn", "model.16.cv2.bn"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for pos, color in zip((1, 7, 11, 21), ("#28689b", "#9b5b38", "#278663", "#9466a3")):
        selected = [next(r for r in gaps if r["layer"] == layer and int(r["batch_position"]) == pos) for layer in layers]
        offset = ((1, 7, 11, 21).index(pos) - 1.5) * .12
        axes[0].scatter([i + offset for i in range(4)], [float(r["D"]) for r in selected], color=color, s=30, label=f"Position {pos}")
    axes[0].axhline(3, color="#555555", linestyle="--", label="Predeclared threshold")
    axes[0].set(xticks=range(4), xticklabels=("Backbone0", "Backbone4 cv1", "Neck16 cv1", "Neck16 cv2"),
                ylabel="Median normalized TRAIN-A mean gap", ylim=(0, 3.3), title="Epoch5 pre-probe buffer screen")
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[0].legend(frameon=False, fontsize=8)
    controls = rows(args.controls)
    # Use the existing low-threshold diagnostic counts, never formal AP.
    selected = [r for r in controls if abs(float(r["confidence"]) - .001) < 1e-12]
    for index, record in enumerate(selected):
        axes[1].bar(index - .16, int(record["box_tp"]), width=.3, color="#28689b", label="Box TP" if index == 0 else None)
        axes[1].bar(index + .16, int(record["mask_tp"]), width=.3, color="#278663", label="Mask TP" if index == 0 else None)
    axes[1].set(xticks=range(len(selected)), xticklabels=[r["case"].replace("_", " ") for r in selected],
                ylabel="True positives / 62 TRAIN instances", title="Same frozen TRAIN batch, conf=.001", ylim=(0, 62))
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].legend(frameon=False)
    fig.savefig(args.output / "e5_statistics_and_train_controls.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
