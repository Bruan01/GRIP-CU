"""SC-CL schematic using SciPilot's publication style, export, and QA helpers.

SciPilot targets quantitative plots; this adapts only its figure-production rules.
Run with the project's guardenv Python from any working directory.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

SKILL_SCRIPTS = Path("/home/kieran/.codex/skills/scipilot-figure-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, render_preview  # noqa: E402


OUT = Path(__file__).resolve().parent
NAVY = "#17324D"
TEAL = "#007F86"
AMBER = "#B67A12"
GRAY = "#677887"
PALE = "#F3F7F8"
TEAL_PALE = "#EAF5F4"
AMBER_PALE = "#FFF5DD"


def box(ax, xy, wh, label, *, edge=NAVY, fill="white", size=7.1, weight="medium"):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.003,rounding_size=0.011",
            facecolor=fill, edgecolor=edge, linewidth=0.9,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=NAVY, fontsize=size, weight=weight, linespacing=1.15)


def arrow(ax, points, *, color=TEAL, lw=1.15, dash=False):
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw,
                linestyle=(0, (3, 2)) if dash else "solid", solid_capstyle="round")
    p, q = points[-2:]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7.5,
                                 color=color, linewidth=lw,
                                 linestyle=(0, (3, 2)) if dash else "solid"))


def main():
    setup_style(journal="nature", lang="en", use_sciplots=False,
                constrained_layout=False)
    fig = plt.figure(figsize=(7.2, 4.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(.034, .956, "A  TRAIN-ONLY CANDIDATE CONSTRUCTION", color=NAVY,
            fontsize=8.2, weight="bold", va="center")
    ax.plot([.034, .966], [.931, .931], color=NAVY, lw=.75)

    box(ax, (.035, .705), (.155, .135), "Training graph\n+ questions", fill=PALE)
    box(ax, (.035, .445), (.155, .105), "Frozen B1\ntrain scores", fill=PALE)
    box(ax, (.275, .738), (.155, .105), "Known-pair filter", edge=TEAL, fill=TEAL_PALE)
    box(ax, (.275, .584), (.155, .105), "Structure profile", edge=TEAL, fill=TEAL_PALE)
    box(ax, (.275, .442), (.155, .105), "B1 score violation", edge=TEAL, fill=TEAL_PALE)
    box(ax, (.535, .592), (.155, .145), "Mixture sampler\nsoftmax + uniform floor",
        edge=TEAL, fill=TEAL_PALE, size=6.8)
    box(ax, (.790, .592), (.175, .145), "FROZEN BANK\nK = 9 per question",
        edge=AMBER, fill=AMBER_PALE, size=7.5, weight="bold")

    arrow(ax, [(.190, .778), (.275, .791)])
    arrow(ax, [(.190, .736), (.225, .736), (.225, .637), (.275, .637)])
    arrow(ax, [(.190, .498), (.275, .495)], color=GRAY)
    arrow(ax, [(.430, .791), (.475, .791), (.475, .705), (.535, .705)])
    arrow(ax, [(.430, .637), (.535, .657)])
    arrow(ax, [(.430, .495), (.488, .495), (.488, .615), (.535, .615)])
    arrow(ax, [(.690, .664), (.790, .664)])
    ax.text(.353, .722, "exclude gold / train-known", color=GRAY,
            fontsize=6.0, ha="center", va="center")
    ax.text(.775, .565, "frozen before Stage 2", color=AMBER,
            fontsize=6.0, ha="center", va="center")

    ax.text(.034, .390, "B  STAGE 2 JOINT TRAINING", color=NAVY,
            fontsize=8.2, weight="bold", va="center")
    ax.plot([.034, .966], [.366, .366], color=NAVY, lw=.75)
    box(ax, (.115, .251), (.205, .080), "Original generation loss",
        edge=GRAY, fill=PALE, size=6.9)
    box(ax, (.475, .192), (.195, .090), "Candidate InfoNCE",
        edge=TEAL, fill=TEAL_PALE, size=7.0)
    box(ax, (.820, .235), (.145, .090), "Trained GRIP", size=7.3, weight="bold")
    # Leave a small optical gap where the bank route crosses the existing loss path.
    ax.plot([.320, .564], [.291, .291], color=GRAY, lw=1.15)
    arrow(ax, [(.576, .291), (.820, .291)], color=GRAY)
    arrow(ax, [(.878, .592), (.878, .430), (.570, .430), (.570, .282)])
    arrow(ax, [(.670, .237), (.749, .237), (.749, .263), (.820, .263)])
    ax.text(.345, .313, "existing path", color=GRAY, fontsize=6.0, ha="left")

    ax.text(.034, .131, "C  EVALUATION ONLY", color=NAVY,
            fontsize=8.2, weight="bold", va="center")
    ax.plot([.034, .966], [.112, .112], color=NAVY, lw=.75)
    box(ax, (.420, .016), (.225, .068), "Free generation EM", size=6.9)
    box(ax, (.688, .016), (.277, .068), "Candidate ranking MRR / Hits", size=6.5)
    arrow(ax, [(.893, .235), (.893, .100), (.532, .100), (.532, .084)], color=GRAY, lw=.9)
    arrow(ax, [(.893, .100), (.826, .100), (.826, .084)], color=GRAY, lw=.9)
    ax.text(.140, .051, "No feedback to bank", color=GRAY, fontsize=6.6,
            ha="center", va="center")
    ax.plot([.292, .292], [.010, .095], color=GRAY, lw=.8, linestyle=(0, (3, 2)))

    preview = OUT / "scipilot_adapted_sccl_preview.png"
    render_preview(fig, str(preview), dpi=180)
    issues = audit_layout(fig)
    for severity, message in issues:
        print(f"{severity}: {message}")
    if any(level == "FAIL" for level, _ in issues):
        raise RuntimeError("SciPilot layout audit failed")
    export_figure(fig, str(OUT / "scipilot_adapted_sccl"),
                  formats=["pdf", "svg", "png"], size_inches=(7.2, 4.4),
                  dpi=600, grayscale_preview=False, tight=False)
    # Keep the helper's exact final dimensions. Its grayscale option redraws the
    # PNG with a tight box, so convert the already exported PNG directly.
    Image.open(OUT / "scipilot_adapted_sccl.png").convert("L").save(
        OUT / "scipilot_adapted_sccl_grayscale.png", dpi=(600, 600))
    plt.close(fig)


if __name__ == "__main__":
    main()
