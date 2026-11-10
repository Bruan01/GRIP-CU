"""Portable Python plots from audited aggregate CSV; no model or raw-bank access.

Uses the project's established Python manuscript-figure workflow. Default inputs
are the accompanying sampling-distribution evidence exports.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image, ImageOps
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
DATA = PAPER / "artifacts/sampling_distribution_seed2026"
DATASETS = ("nell23k", "codexm", "fb15k237_2", "wn18rr")
NAMES = {"nell23k": "NELL23K", "codexm": "CoDEx-M", "fb15k237_2": "FB15K237-2", "wn18rr": "WN18RR"}
LABELS = {"train_edges": "Training edges", "eligible_gold": "Eligible QA gold",
          "full": "Full SC-CL", "uniform_observed": "Uniform (observed)",
          "quota_observed": "Quota-matched", "legal_uniform_expected": "Uniform (expected)",
          "profile_only_observed": "Profile only", "violation_only_observed": "Violation only"}
COLORS = {"train_edges": "#44494D", "eligible_gold": "#B39A70", "full": "#427C96",
          "uniform_observed": "#899795", "quota_observed": "#9B859E", "legal_uniform_expected": "#687578",
          "profile_only_observed": "#B5A796", "violation_only_observed": "#9DAFC0"}
STYLES = {"train_edges": "-", "eligible_gold": (0, (5, 2)), "full": "-",
          "uniform_observed": (0, (4, 2)), "quota_observed": (0, (2, 2)),
          "legal_uniform_expected": (0, (1, 2))}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.6,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .65,
    "lines.linewidth": 1.25, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
    "savefig.facecolor": "white", "svg.hashsalt": "sccl_sampling_distribution_seed2026"})


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decorate(ax, letter, title):
    ax.set_title(title, loc="left", pad=9)
    ax.text(-.12, 1.06, letter, transform=ax.transAxes, weight="bold", fontsize=9, va="bottom")
    ax.tick_params(length=3, width=.6)
    ax.grid(axis="y", color="#ECEEF0", lw=.5)
    ax.set_axisbelow(True)


def legend_handles(methods):
    return [Line2D([0], [0], color=COLORS[m], lw=1.8 if m == "full" else 1.25,
                   linestyle=STYLES.get(m, "-"), label=LABELS[m]) for m in methods]


def cdf(ax, dataset, rel, include_gold=True):
    values = rel[dataset]
    n = len(values["train_edges"])
    x = np.r_[0, np.arange(1, n + 1) / n * 100]
    methods = ["train_edges", "eligible_gold", "uniform_observed", "quota_observed", "legal_uniform_expected", "full"]
    for method in methods:
        if method not in values or (method == "eligible_gold" and not include_gold):
            continue
        if method == "legal_uniform_expected" and "uniform_observed" in values:
            continue
        y = np.r_[0, np.cumsum(values[method]) * 100]
        ax.plot(x, y, color=COLORS[method], linestyle=STYLES[method],
                lw=1.9 if method == "full" else 1.15, zorder=4 if method == "full" else 2)
    ax.set(xlim=(0, 100), ylim=(0, 101), xlabel="Training-frequency ranked relations (%)", ylabel="Cumulative probability (%)")
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])


def distance_bars(ax, dataset, metrics, all_controls=False):
    order = ["full", "uniform_observed", "quota_observed"]
    if all_controls:
        order += ["profile_only_observed", "violation_only_observed"]
    order += ["legal_uniform_expected"]
    order = [m for m in order if m in metrics[dataset]]
    y = np.arange(len(order))
    x = [float(metrics[dataset][m]["js_divergence_bits"]) for m in order]
    for i, (method, value) in enumerate(zip(order, x)):
        expected = method == "legal_uniform_expected"
        ax.barh(i, value, height=.59, color="white" if expected else COLORS[method],
                edgecolor=COLORS[method], lw=.8, hatch="///" if expected else None)
        ax.text(value + .008, i, f"{value:.3f}", va="center", fontsize=6.6)
    ax.set_yticks(y, [LABELS[m] for m in order]); ax.invert_yaxis()
    ax.set(xlim=(0, .49), xlabel="Jensen-Shannon divergence (bits)")
    ax.set_xticks([0, .1, .2, .3, .4]); ax.grid(False)
    ax.grid(axis="x", color="#ECEEF0", lw=.5)


def export(fig, name, pdfbook, qa):
    fig.canvas.draw()
    outside = []
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        bb = text.get_window_extent(renderer)
        if bb.width > 0 and bb.height > 0 and (bb.x0 < -2 or bb.y0 < -2 or bb.x1 > fig.bbox.width + 2 or bb.y1 > fig.bbox.height + 2):
            outside.append(text.get_text())
    assert not outside, (name, outside)
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        if suffix in ("svg", "pdf"):
            kwargs["metadata"] = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(HERE / f"{name}.{suffix}", **kwargs)
    pdfbook.savefig(fig)
    svg = ET.parse(HERE / f"{name}.svg")
    editable = len(svg.findall(".//{http://www.w3.org/2000/svg}text"))
    assert editable > 10
    pdf = PdfReader(HERE / f"{name}.pdf")
    assert len(pdf.pages) == 1
    if any(k in name for k in DATASETS):
        assert NAMES[next(k for k in DATASETS if k in name)] in pdf.pages[0].extract_text()
    with Image.open(HERE / f"{name}.png") as im:
        size = im.size
    qa[name] = {"svg_editable_text_nodes": editable, "png_pixels": size,
                "width_mm": fig.get_figwidth() * 25.4, "height_mm": fig.get_figheight() * 25.4,
                "text_outside_canvas": outside,
                "sha256": {s: sha(HERE / f"{name}.{s}") for s in ("svg", "pdf", "png")}}
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    source = args.data_dir
    summary = json.loads((source / "summary.json").read_text())
    rrows = csv_rows(source / "source_data/relation_distribution.csv")
    brows = csv_rows(source / "source_data/frequency_bin_distribution.csv")
    crows = csv_rows(source / "source_data/conditional_frequency_distribution.csv")
    mrows = csv_rows(source / "source_data/distribution_metrics.csv")
    rel = defaultdict(dict)
    for dataset in DATASETS:
        for method in {r["method"] for r in rrows if r["dataset"] == dataset}:
            rows = sorted([r for r in rrows if r["dataset"] == dataset and r["method"] == method], key=lambda r: int(r["train_rank"]))
            rel[dataset][method] = np.array([float(r["probability"]) for r in rows])
    metrics = defaultdict(dict)
    for r in mrows:
        if r["reference"] == "train_edges":
            metrics[r["dataset"]][r["method"]] = r
    qa = {}
    with PdfPages(HERE / "sampling_distributions_all.pdf", metadata={"Title": "SC-CL sampling distributions across four datasets", "CreationDate": None, "ModDate": None}) as pdfbook:
        fig, axs = plt.subplots(2, 2, figsize=(183 / 25.4, 146 / 25.4))
        fig.subplots_adjust(left=.09, right=.98, bottom=.15, top=.88, hspace=.55, wspace=.36)
        fig.suptitle("Training-edge, QA-target and negative-sampling distributions", fontsize=9, y=.97)
        for i, (dataset, ax) in enumerate(zip(DATASETS, axs.flat)):
            s = summary[dataset]; cdf(ax, dataset, rel)
            decorate(ax, "abcd"[i], f"{NAMES[dataset]}  |  R={s['train_relations']}; n={s['eligible_questions']:,}")
        handles = legend_handles(["train_edges", "eligible_gold", "full", "uniform_observed", "quota_observed", "legal_uniform_expected"])
        fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.52, .035), columnspacing=1.5, handlelength=2.6)
        fig.text(.5, .014, "Relations ranked by training-edge counts. CoDEx-M has no trained controls; its dotted Uniform line is analytic.", ha="center", fontsize=6.2, color="#555B60")
        export(fig, "fig01_cumulative_four_datasets", pdfbook, qa)

        fig, axs = plt.subplots(2, 2, figsize=(183 / 25.4, 133 / 25.4))
        fig.subplots_adjust(left=.19, right=.98, bottom=.12, top=.88, hspace=.68, wspace=.95)
        fig.suptitle("Distance from original training-edge distribution", fontsize=9, y=.97)
        for i, (dataset, ax) in enumerate(zip(DATASETS, axs.flat)):
            distance_bars(ax, dataset, metrics, all_controls=True)
            decorate(ax, "abcd"[i], NAMES[dataset])
        fig.text(.5, .035, "Lower divergence means closer marginals; it is not an accuracy metric. Hatched bars are analytic expectations.", ha="center", fontsize=6.2, color="#555B60")
        export(fig, "fig02_distribution_distance", pdfbook, qa)

        cmap = colors.LinearSegmentedColormap.from_list("muted_signed", ["#B58D61", "#FAFAF8", "#477D95"])
        cmap.set_bad("#E8EAEC")
        for dataset in DATASETS:
            s = summary[dataset]
            fig, axs = plt.subplots(2, 2, figsize=(183 / 25.4, 154 / 25.4))
            fig.subplots_adjust(left=.14, right=.97, bottom=.145, top=.89, hspace=.53, wspace=.95)
            fig.suptitle(f"{NAMES[dataset]}: original graph, QA targets and frozen negative candidates", fontsize=9, y=.97)
            fig.text(.5, .93, f"{s['train_edges']:,} train edges | {s['train_relations']} relation labels | {s['eligible_questions']:,} eligible QA | {s['negative_slots']:,} negative slots", ha="center", fontsize=7)
            ax = axs[0, 0]; cdf(ax, dataset, rel); decorate(ax, "a", "Marginal concentration")

            ax = axs[0, 1]
            data = [r for r in brows if r["dataset"] == dataset]
            bins = sorted({int(r["frequency_bin"]) for r in data})
            for j, method in enumerate(("train_edges", "eligible_gold", "full")):
                lookup = {int(r["frequency_bin"]): 100 * float(r["probability"]) for r in data if r["method"] == method}
                ax.bar(np.arange(len(bins)) + (j - 1) * .25, [lookup[b] for b in bins], width=.24,
                       color=COLORS[method], alpha=.88, edgecolor="white", lw=.15)
            stride = max(1, len(bins) // 8)
            ticks = np.arange(len(bins))[::stride]
            ax.set_xticks(ticks, [str(bins[i]) for i in ticks])
            ax.set(xlabel="Relation-frequency bin: floor(log2 train edges)", ylabel="Probability mass (%)")
            decorate(ax, "b", "Frequency composition")

            ax = axs[1, 0]
            mats = {}
            for method in ("full", "legal_uniform_expected"):
                lookup = {(int(r["gold_frequency_bin"]), int(r["negative_frequency_bin"])): r["conditional_probability"] for r in crows if r["dataset"] == dataset and r["method"] == method}
                mats[method] = np.array([[float(lookup[(gb, nb)]) if lookup[(gb, nb)] else np.nan for nb in bins] for gb in bins])
            delta = (mats["full"] - mats["legal_uniform_expected"]) * 100
            limit = max(.1, float(np.nanmax(np.abs(delta))))
            im = ax.imshow(np.ma.masked_invalid(delta), origin="lower", cmap=cmap,
                           norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto", interpolation="nearest")
            row_n = {int(r["gold_frequency_bin"]): round(float(r["row_slots"]) / 9) for r in crows if r["dataset"] == dataset and r["method"] == "full"}
            ax.set_xticks(ticks, [str(bins[i]) for i in ticks])
            ax.set_yticks(np.arange(len(bins)), [f"{b} ({row_n[b]})" for b in bins], fontsize=5.8)
            ax.set(xlabel="Negative relation frequency bin", ylabel="Gold bin (eligible QA n)")
            # Signs keep the direction of stronger displayed shifts readable in gray.
            for y, x in np.argwhere(np.isfinite(delta) & (np.abs(delta) >= .35 * limit)):
                ax.text(x, y, "+" if delta[y, x] > 0 else "−", ha="center", va="center", fontsize=5, color="#343A3E")
            decorate(ax, "c", "Conditional shift: Full minus Uniform")
            ax.grid(False)
            cbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.025)
            cbar.ax.tick_params(labelsize=6, length=2); cbar.set_label("Row probability difference (pp)", fontsize=6.5)

            ax = axs[1, 1]; distance_bars(ax, dataset, metrics, all_controls=True)
            decorate(ax, "d", "Distance to training-edge marginals")
            visible = [m for m in ("train_edges", "eligible_gold", "full", "uniform_observed", "quota_observed", "legal_uniform_expected")
                       if m in rel[dataset] and not (m == "legal_uniform_expected" and "uniform_observed" in rel[dataset])]
            fig.legend(handles=legend_handles(visible), ncol=2 if len(visible) == 4 else 3,
                       loc="lower center", bbox_to_anchor=(.53, .036), columnspacing=1.2, handlelength=2.5)
            note = "Uniform expectation uses each eligible question's legal pool; gray heatmap rows have no observed gold QA."
            if dataset == "wn18rr":
                note = "WN: 55 historical training pairs unverified; retain parent masks. Conditional baseline is the analytic legal-pool Uniform."
            elif dataset == "codexm":
                note = "CoDEx-M: no trained Uniform/quota controls. Dotted lines and hatched bars are analytic legal-pool expectations."
            fig.text(.5, .013, note, ha="center", fontsize=6.0, color="#555B60")
            export(fig, f"distribution_{dataset}", pdfbook, qa)
    book = PdfReader(HERE / "sampling_distributions_all.pdf")
    assert len(book.pages) == 6
    # Selected Python backend also produces the review contact sheet and grayscale check.
    thumbs = []
    for dataset in DATASETS:
        with Image.open(HERE / f"distribution_{dataset}.png") as im:
            thumb = im.convert("RGB"); thumb.thumbnail((1100, 1000)); thumbs.append(thumb.copy())
    cellw, cellh = max(i.width for i in thumbs), max(i.height for i in thumbs)
    sheet = Image.new("RGB", (cellw * 2, cellh * 2), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 2) * cellw, (i // 2) * cellh))
    sheet.save(HERE / "overview_four_datasets.png")
    ImageOps.grayscale(sheet).save(HERE / "overview_four_datasets_grayscale.png")
    qa["scope"] = "Python-only generation and export; all panels from frozen-bank aggregate CSV; no new sampling or training"
    qa["combined_pdf_pages"] = len(book.pages)
    qa["source_data_sha256"] = {p.name: sha(p) for p in sorted((source / "source_data").glob("*.csv"))}
    qa["render_inspection"] = "pending visual review"
    (HERE / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n")
    print("Exported six figures, editable SVGs, PDFs, PNGs and six-page PDF book.")


if __name__ == "__main__":
    main()
