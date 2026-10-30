"""Preset cross-dataset effect-size plot for the SC-CL manuscript.

The values are placeholders copied from the manuscript's extended table.
Formal runs should replace PRESET_DELTAS and keep the export contract.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PRESET_DELTAS = {
    "NELL23K": 1.9,
    "FB-local": 2.2,
    "WN18RR": 0.9,
    "FB-canonical": 2.2,
    "YAGO3-10": 1.7,
    "CoDEx-S": 1.5,
    "CoDEx-M": 2.2,
}


def main() -> None:
    labels = list(PRESET_DELTAS)
    values = np.asarray(list(PRESET_DELTAS.values()))
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.0, 2.9), dpi=180)
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color="#8f1d35",
        edgecolor="#252525",
        linewidth=0.65,
        hatch="///",
    )
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=7)
    ax.axhline(0.0, color="#252525", linewidth=0.8)
    ax.set_ylabel("MRR gain (pt)", fontsize=9)
    ax.set_xticks(x, labels, rotation=28, ha="right", fontsize=7.5)
    ax.set_ylim(0, 2.7)
    ax.set_yticks([0, 1, 2])
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.6)

    output = Path(__file__).resolve().parents[1] / "dataset_delta_placeholder.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
