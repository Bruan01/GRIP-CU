"""Publication-style placeholder plot for the SC-CL paper.

This follows the grouped/hatch layout of the bundled paper-plot bar template.
The values are deliberately kept in one named constant so formal runs can
replace them without changing the plotting code or the manuscript narrative.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PRESET_VALUES = {
    "NELL23K": {"B1": 84.2, "Random-9": 86.2, "SC-CL-9": 88.1},
    "FB15K237-2": {"B1": 73.5, "Random-9": 75.1, "SC-CL-9": 77.3},
}


def main() -> None:
    datasets = list(PRESET_VALUES)
    methods = ["B1", "Random-9", "SC-CL-9"]
    colors = ["#8d99ae", "#b8c0cc", "#8f1d35"]
    hatches = ["", "", "///"]
    x = np.arange(len(datasets))
    width = 0.22

    fig, ax = plt.subplots(figsize=(5.7, 3.1), dpi=180)
    for idx, method in enumerate(methods):
        values = [PRESET_VALUES[d][method] for d in datasets]
        bars = ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            label=method,
            color=colors[idx],
            edgecolor="#252525",
            linewidth=0.65,
            hatch=hatches[idx],
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)

    ax.set_ylabel("MRR (%)", fontsize=9)
    ax.set_xticks(x, datasets, fontsize=8)
    ax.set_ylim(68, 90.5)
    ax.set_yticks([70, 75, 80, 85, 90])
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7.5)
    fig.tight_layout(pad=0.6)

    output = Path(__file__).resolve().parents[1] / "mrr_placeholder.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
