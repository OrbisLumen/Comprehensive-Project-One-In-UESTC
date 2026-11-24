"""Plot validation bends selected using only the expert's steering."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_predictions(path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if {r["route"] for r in rows} != {"trace6"}:
        raise ValueError("Expected trace6 validation predictions")
    return tuple(np.array([float(r[key]) for r in rows]) for key in ("frame", "target", "prediction"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course", type=Path)
    parser.add_argument("resnet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frames, target, course = read_predictions(args.course)
    old_frames, old_target, resnet = read_predictions(args.resnet)
    if (not np.array_equal(frames, old_frames) or not np.allclose(target, old_target)
            or not np.all(np.diff(frames) == 1)):
        raise ValueError("Expected the same consecutive frames and labels in both files")

    # Select centers with room for context on both sides, without using model predictions.
    mean = np.convolve(target, np.ones(21) / 21, mode="same")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharey=True, layout="constrained")
    for ax, sign, title in zip(axes, (-1, 1), ("Left bend", "Right bend")):
        center = 30 + np.argmax(sign * mean[30:-30])
        window = slice(center - 30, center + 31)
        ax.axvspan(frames[center-10], frames[center+10], color="#f3f0e6", zorder=0)
        for values, label, color, width in [
            (target, "Built-in driver", "#29333b", 1.6),
            (resnet, "ResNet", "#ba660c", 1.5),
            (course, "AutoDriveNet", "#1576a1", 1.6),
        ]:
            ax.plot(frames[window], values[window], label=label, color=color, lw=width)
        ax.axhline(0, color="#a9adb0", linewidth=0.65)
        ax.set(title=f"{title}  |  trace6", ylabel="Steering", ylim=(-1.1, 1.1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=3, loc="upper left", fontsize=9)
    axes[-1].set_xlabel("Recorded frame")
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
