"""Observed candidate-ranking MRR figure for the SC-CL manuscript.

The layout follows the bundled grouped/hatch paper-plot template.  Values are
copied from the audited result summaries, not from the former preset figure:
NELL23K is the strict prefix-200 paired comparison, while FB15K237-2 is the
historical full candidate-score comparison and is labelled accordingly.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    # The WSL XeLaTeX manuscript compiles independently; the plotting host
    # lacks Matplotlib's optional pdfLaTeX type1ec package, so use the local
    # serif fallback while retaining the bundled layout and hatch style.
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "hatch.color": "white",
    "hatch.linewidth": 1.4,
})

COLORS = ["#D3D3D3", "#D00000"]
HATCHES = ["", "//"]

PANELS = [
    {
        "title": r"(a) NELL23K: paired prefix-200",
        "labels": ["B1", "SC-CL"],
        "values": [93.36, 94.96],
        "best": "SC-CL",
    },
    {
        "title": r"(b) FB15K237-2: historical full",
        "labels": ["B1", "InfoNCE-i0"],
        "values": [88.01, 95.41],
        "best": "InfoNCE-i0",
    },
]


def draw_panel(ax, panel):
    x = np.array([0.0])
    labels = panel["labels"]
    values = panel["values"]
    width = 0.34
    for idx, (label, value) in enumerate(zip(labels, values)):
        offset = (idx - (len(labels) - 1) / 2) * width
        bar = ax.bar(
            x + offset,
            [value],
            width=width * 0.92,
            color=COLORS[idx],
            hatch=HATCHES[idx],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
            label=label,
        )[0]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold" if label == panel["best"] else "normal",
            color="black",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["candidate MRR"], fontsize=10.2)
    ax.set_ylabel(r"MRR (\%)", fontsize=11.2)
    ax.set_ylim(85, 100)
    ax.set_xlim(-0.62, 0.62)
    ax.set_title(panel["title"], fontsize=12.6, pad=6)
    ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_linewidth(0.9)
            spine.set_color("#333333")
            spine.set_zorder(4)
    ax.tick_params(length=0, labelsize=9.8)

    handles = [
        mpatches.Patch(
            facecolor=COLORS[idx],
            hatch=HATCHES[idx],
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        for idx, label in enumerate(labels)
    ]
    legend = ax.legend(
        handles=handles,
        fontsize=8.7,
        loc="upper left",
        framealpha=1.0,
        facecolor="white",
        edgecolor="#C8C8C8",
        fancybox=False,
        borderpad=0.28,
        labelspacing=0.26,
        handlelength=1.7,
        handletextpad=0.45,
        borderaxespad=0.28,
    )
    for text in legend.get_texts():
        if text.get_text() == panel["best"]:
            text.set_fontweight("bold")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.65))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.83, wspace=0.28)
    for ax, panel in zip(axes, PANELS):
        draw_panel(ax, panel)
    fig.text(0.5, 0.02, "Higher is better; values are percentages.", ha="center", fontsize=9.2)

    output = Path(__file__).resolve().parents[1] / "observed_candidate_mrr.png"
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
