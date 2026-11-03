"""Build auditable paper figures with Python; default input is portable CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image, ImageOps
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = HERE / "source_data"
B1, SC, GAIN, DROP = "#848991", "#356A91", "#307F71", "#B36449"
LABELS = {"nell23k": "NELL23K", "codexm": "CoDEx-M"}
DATASETS = tuple(LABELS)
SOURCE_RECORDS: list[dict] = []
plt.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 8,
    "axes.titleweight": "bold", "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "legend.frameon": False, "legend.fontsize": 7,
    "savefig.facecolor": "white", "hatch.linewidth": .5,
    "svg.hashsalt": "sccl-python-results-20261103",
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, role, expected=None):
    digest = sha(path)
    require(expected is None or expected == digest, f"Source digest changed: {role}")
    SOURCE_RECORDS.append({"evidence_id": role, "sha256": digest,
                           "matches_prior_audit": None if expected is None else True})


def read_jsonl(path, role, expected=None):
    record(path, role, expected)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(name, rows):
    require(bool(rows), f"Empty data: {name}")
    DATA.mkdir(exist_ok=True)
    with (DATA / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name):
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aligned(left, right):
    for key in ("question_id", "split", "question", "target"):
        require(left[key] == right[key], f"Pair mismatch in {key}")
    require(left["split"] == "test", "Non-test question")


def refresh_sources():
    spec = importlib.util.spec_from_file_location(
        "answer_normalization", ROOT / "13_base_method/grip-exp/evaluation/utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normalize = module.normalize_answer
    direct = []
    for dataset, day in (("nell23k", "20261102"), ("codexm", "20261103")):
        evidence = ROOT / f"11_paper/artifacts/{dataset}_single_seed_test_{day}.json"
        record(evidence, f"{dataset}:verified_snapshot")
        audit = json.loads(evidence.read_text())
        require(audit["seed"] == 2026, "Unexpected training seed")
        models = {}
        for original, expected in audit["evidence_sha256"].items():
            if original.endswith("predictions.jsonl"):
                relative = original.split("/08_experiments/", 1)[1]
                role = "selected" if "/selected_test/" in original else "baseline"
                rows = read_jsonl(ROOT / "08_experiments" / relative,
                                  f"{dataset}:{role}:predictions", expected)
                require(len(rows) == audit["test_count"], "Incomplete direct predictions")
                by_id = {row["question_id"]: row for row in rows}
                require(len(by_id) == len(rows), "Duplicate question ID")
                for row in rows:
                    em = any(normalize(str(row["response"])) == normalize(str(t))
                             for t in row["target"])
                    require(type(row["correct"]) is bool and em == row["correct"],
                            "Exact-match flag differs from recomputed response")
                models[role] = by_id
        baseline, selected = models["baseline"], models["selected"]
        require(baseline.keys() == selected.keys(), "Unmatched direct cohort")
        outcomes = Counter()
        for qid, left in baseline.items():
            right = selected[qid]
            aligned(left, right)
            b, s = int(left["correct"]), int(right["correct"])
            outcomes[(b, s)] += 1
            direct.append({"dataset": dataset, "question_id": qid,
                           "baseline_correct": b, "selected_correct": s})
        require(sum(r["correct"] for r in baseline.values()) == audit["baseline_correct"],
                "Baseline total differs from audit")
        require(sum(r["correct"] for r in selected.values()) == audit["selected_correct"],
                "Selected total differs from audit")
        for cell, key in (((0, 0), "both_wrong"), ((0, 1), "rescue"),
                          ((1, 0), "regression"), ((1, 1), "both_correct")):
            require(outcomes[cell] == audit["paired_counts"][key], "Paired count differs from audit")
    write_csv("direct_pairs.csv", direct)

    base = ROOT / "08_experiments/Hard_Negative_Contrastive_GRIP/results/candidate_rerank"
    baseline = read_jsonl(base / "nell23k_baseline_7b_pilot200/predictions.jsonl",
                         "historical_ranking:baseline:prefix200")
    selected = read_jsonl(base / "nell23k_structure_counterfactual_06_full/predictions.jsonl",
                         "historical_ranking:sccl:full")
    require(len(baseline) == 200 and len(selected) == 4943, "Unexpected ranking cohort size")
    require(len({r['question_id'] for r in selected}) == 4943, "Duplicate ranking question ID")
    for left, right in zip(baseline, selected[:200]):
        aligned(left, right)
        require([c['candidate'] for c in left['candidate_scores']] ==
                [c['candidate'] for c in right['candidate_scores']], "Candidate order differs")
    for row in [*baseline, *selected]:
        candidates = sorted(row["candidate_scores"], key=lambda c: -c["score"])
        targets = {normalize(t) for t in row["target"]}
        rank = next(i for i, c in enumerate(candidates, 1) if normalize(c['candidate']) in targets)
        require(rank == row['gold_rank'], "Gold rank differs from candidate scores")
        require(bool(row['correct']) == (rank == 1), "Rank and correctness disagree")
    write_csv("ranking_records.csv", [
        {"question_id": row['question_id'], "selected_rank": row['gold_rank'],
         "baseline_rank": baseline[i]['gold_rank'] if i < 200 else ""}
        for i, row in enumerate(selected)])
    (DATA / "provenance.json").write_text(json.dumps({
        "schema_version": 1, "seed": 2026, "synthetic_data": False,
        "source_audit": SOURCE_RECORDS, "direct_flag_checks": 30508,
        "paired_direct_questions": 15254, "ranking_score_checks": 5143,
        "prompt_target_split_alignment": True,
        "ranking_prefix_candidate_order_alignment": True,
        "scope": "Local single-seed observations; tuned configurations; historical ranking is a separate run",
    }, indent=2) + "\n")


def summarize():
    pairs = read_csv("direct_pairs.csv")
    require(len(pairs) == 15254, "Direct cohort changed")
    metrics, transitions = [], []
    for dataset in DATASETS:
        rows = [r for r in pairs if r['dataset'] == dataset]
        require(len({r['question_id'] for r in rows}) == len(rows), "Duplicate portable question ID")
        require(all(r['baseline_correct'] in ('0', '1') and r['selected_correct'] in ('0', '1')
                    for r in rows), "Invalid correctness value")
        counts = Counter((int(r['baseline_correct']), int(r['selected_correct'])) for r in rows)
        n = len(rows)
        b = counts[(1, 0)] + counts[(1, 1)]
        s = counts[(0, 1)] + counts[(1, 1)]
        metrics.append({"dataset": dataset, "n": n, "baseline_correct": b,
                        "selected_correct": s, "baseline_em_percent": 100*b/n,
                        "selected_em_percent": 100*s/n, "net_correct": s-b,
                        "delta_pp": 100*(s-b)/n, "seed": 2026})
        for (bval, sval), name in (((0, 0), "both_wrong"), ((0, 1), "rescue"),
                                  ((1, 0), "regression"), ((1, 1), "both_correct")):
            transitions.append({"dataset": dataset, "outcome": name,
                                "baseline_correct": bval, "selected_correct": sval,
                                "count": counts[(bval, sval)], "n": n,
                                "percent": 100*counts[(bval, sval)]/n})
        require(s-b == counts[(0, 1)]-counts[(1, 0)], "Net change identity failed")
    write_csv("direct_metrics.csv", metrics)
    write_csv("paired_outcomes.csv", transitions)
    records = read_csv("ranking_records.csv")
    require(len(records) == 4943 and sum(bool(r['baseline_rank']) for r in records) == 200,
            "Ranking cohort changed")
    require(all(r['baseline_rank'] for r in records[:200]), "Baseline no longer matches prefix")
    ranking = []
    for scope, model, ranks in (
        ("prefix200", "B1", [int(r['baseline_rank']) for r in records[:200]]),
        ("prefix200", "SC-CL", [int(r['selected_rank']) for r in records[:200]]),
        ("full", "SC-CL", [int(r['selected_rank']) for r in records]),
    ):
        require(all(1 <= r <= 10 for r in ranks), "Rank outside ten candidates")
        a = np.asarray(ranks)
        for metric, value in (("H@1", np.mean(a <= 1)), ("MRR", np.mean(1/a)),
                              ("H@3", np.mean(a <= 3))):
            ranking.append({"scope": scope, "model": model, "metric": metric,
                            "n": len(a), "value_percent": 100*float(value), "seed": 2026})
    write_csv("ranking_metrics.csv", ranking)
    return metrics, transitions, ranking


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=12)
    ax.text(-.12, 1.055, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)


def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E6E8EB', linewidth=.5)
    ax.tick_params(length=2.5, width=.6)


def figure_direct(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 96/25.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.085, right=.98, bottom=.25, top=.78, wspace=.4)
    ax, delta = axes
    x = np.arange(2)
    for shift, key, color, label, hatch in (
        (-.18, 'baseline_em_percent', B1, 'Matched B1', ''),
        (.18, 'selected_em_percent', SC, 'Stage-2 SC-CL (tuned)', '//')):
        vals = [r[key] for r in metrics]
        ax.bar(x+shift, vals, width=.31, color=color, label=label, hatch=hatch,
               edgecolor='white', linewidth=.6)
        for xpos, val in zip(x+shift, vals):
            ax.text(xpos, val+1.8, f'{val:.2f}', ha='center', fontsize=6.7)
    ax.set_xticks(x, [f"{LABELS[r['dataset']]}\n$n$ = {r['n']:,}" for r in metrics])
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Direct-decoding EM (%)')
    panel(ax, 'a', 'Full-test direct decoding')
    grid(ax)
    gains = [r['delta_pp'] for r in metrics]
    delta.bar(x, gains, width=.48, color=SC, edgecolor=SC, linewidth=.6)
    for i, r in enumerate(metrics):
        delta.text(i, gains[i]+.12, f"+{gains[i]:.2f} pp\n+{r['net_correct']} correct",
                   ha='center', va='bottom', fontsize=7.2, color=SC, fontweight='bold')
    delta.set_ylim(0, 4.4)
    delta.set_yticks([0, 1, 2, 3, 4])
    delta.set_xticks(x, [LABELS[r['dataset']] for r in metrics])
    delta.set_ylabel('EM change vs matched B1 (pp)')
    panel(delta, 'b', 'Paired net change')
    grid(delta)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.51, .985))
    fig.text(.5, .075, 'One training seed (2026); validation-selected configurations; no seed-level error bars.',
             ha='center', fontsize=6.8, color='#555555')
    fig.text(.5, .025, 'Local protocol comparison; tuning and optimizer differences prevent single-component attribution.',
             ha='center', fontsize=6.5, color='#555555')
    return fig


def figure_pairs(metrics, transitions):
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 104/25.4))
    fig.subplots_adjust(left=.13, right=.98, bottom=.24, top=.83, wspace=.5)
    colors = {'both_wrong': '#F1F2F4', 'rescue': '#E2F0EC',
              'regression': '#F8E9E1', 'both_correct': '#E3EBF3'}
    names = {'both_wrong': 'Both wrong', 'rescue': 'Rescue (+)',
             'regression': 'Regression (−)', 'both_correct': 'Both correct'}
    for ax, r, letter in zip(axes, metrics, 'ab'):
        panel(ax, letter, f"{LABELS[r['dataset']]} | $n$ = {r['n']:,}")
        rows = [row for row in transitions if row['dataset'] == r['dataset']]
        for cell in rows:
            col, row = cell['selected_correct'], cell['baseline_correct']
            name = cell['outcome']
            ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=colors[name],
                                   edgecolor='white', linewidth=3))
            color = GAIN if name == 'rescue' else DROP if name == 'regression' else '#333333'
            ax.text(col, row-.23, names[name], ha='center', va='center', fontsize=7, color=color)
            ax.text(col, row+.015, f"{cell['count']:,}", ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)
            ax.text(col, row+.25, f"{cell['percent']:.2f}%", ha='center', va='center', fontsize=7)
        ax.set(xlim=(-.5, 1.5), ylim=(1.5, -.5), aspect='equal')
        ax.set_xticks([0, 1], ['Wrong', 'Correct'])
        ax.set_yticks([0, 1], ['Wrong', 'Correct'])
        ax.set_xlabel('Stage-2 SC-CL prediction')
        ax.set_ylabel('Matched B1 prediction', labelpad=5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.5, .115, 'Each cell: number of test questions and percentage of that dataset’s full test cohort.',
             ha='center', fontsize=6.8)
    fig.text(.5, .06, 'Rescue − regression = net gain: 321 − 162 = 159 (NELL23K); 301 − 70 = 231 (CoDEx-M).',
             ha='center', fontsize=6.8)
    fig.text(.5, .018, 'Single seed (2026); same prompts, candidate order, targets and question IDs within each dataset.',
             ha='center', fontsize=6.5, color='#555555')
    return fig


def figure_ranking(ranking):
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 96/25.4))
    fig.subplots_adjust(left=.085, right=.98, bottom=.26, top=.78, wspace=.35)
    x, categories = np.arange(3), ['H@1', 'MRR', 'H@3']
    for model, shift, color, hatch in [('B1', -.18, B1, ''), ('SC-CL', .18, SC, '//')]:
        vals = [next(r['value_percent'] for r in ranking if r['scope'] == 'prefix200'
                     and r['model'] == model and r['metric'] == metric) for metric in categories]
        axes[0].bar(x+shift, vals, width=.31, color=color, label=model,
                    edgecolor='white', linewidth=.6, hatch=hatch)
        for xpos, val in zip(x+shift, vals):
            axes[0].text(xpos, val+2, f'{val:.2f}', ha='center', fontsize=6.4)
    vals = [next(r['value_percent'] for r in ranking if r['scope'] == 'full'
                 and r['metric'] == metric) for metric in categories]
    axes[1].bar(x, vals, width=.44, color=SC, edgecolor='white', hatch='//', linewidth=.6)
    for xpos, val in zip(x, vals):
        axes[1].text(xpos, val+2, f'{val:.2f}', ha='center', fontsize=7)
    for ax in axes:
        ax.set_xticks(x, categories)
        ax.set_ylim(0, 109)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel('Candidate-ranking metric (%)')
        grid(ax)
    panel(axes[0], 'a', 'Paired prefix | $n$ = 200')
    panel(axes[1], 'b', 'SC-CL full test | $n$ = 4,943')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc='upper center',
               bbox_to_anchor=(.5, .985))
    fig.text(.5, .12, 'Historical NELL23K run; teacher-forced scoring of the same ten candidates; seed 2026.',
             ha='center', fontsize=6.8)
    fig.text(.5, .065, 'Prefix follows test-file order; full-test B1 reranking is unavailable; training budgets differ.',
             ha='center', fontsize=6.6, color='#555555')
    fig.text(.5, .018, 'Ranking models and baseline differ from those used for the latest direct-decoding evaluation.',
             ha='center', fontsize=6.5, color='#555555')
    return fig


def export(fig, name, pdf):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text().strip():
            box = artist.get_window_extent(renderer)
            if box.width and box.height and not (
                box.x0 >= -1 and box.y0 >= -1 and
                box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1):
                outside.append(artist.get_text())
    require(not outside, f"Text outside canvas in {name}: {outside}")
    for fmt, dpi in [('svg', 300), ('pdf', 300), ('png', 300), ('tiff', 600)]:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if fmt == 'tiff' else {}
        fig.savefig(HERE / f'{name}.{fmt}', dpi=dpi, **kw)
    pdf.savefig(fig)
    svg = ET.parse(HERE / f'{name}.svg').getroot()
    text_count = sum(el.tag.endswith('}text') for el in svg.iter())
    require(text_count > 10, 'SVG lacks editable text')
    reader = PdfReader(HERE / f'{name}.pdf')
    require(len(reader.pages) == 1 and len(reader.pages[0].extract_text()) > 150,
            'PDF text extraction failed')
    with Image.open(HERE / f'{name}.tiff') as img:
        require(all(abs(v-600) < 1 for v in img.info['dpi']), 'TIFF is not 600 dpi')
        tiff_size = list(img.size)
    fonts = reader.pages[0]['/Resources']['/Font'].get_object()
    font_types = [str(font.get_object()['/Subtype']) for font in fonts.values()]
    require('/Type3' not in font_types, 'PDF contains noneditable Type3 font')
    size = fig.get_size_inches()*25.4
    plt.close(fig)
    return {'figure': name, 'width_mm': round(size[0], 2), 'height_mm': round(size[1], 2),
            'text_outside_canvas': outside, 'editable_svg_text_elements': text_count,
            'pdf_text_extractable': True, 'pdf_font_types': font_types,
            'tiff_dpi': 600, 'tiff_pixels': tiff_size,
            'exports': {ext: sha(HERE/f'{name}.{ext}') for ext in ['svg', 'pdf', 'png', 'tiff']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-source', action='store_true',
                        help='Re-audit local original predictions and refresh portable CSV input')
    args = parser.parse_args()
    if args.refresh_source:
        refresh_sources()
    metrics, transitions, ranking = summarize()
    specs = [('fig01_direct_decoding', figure_direct, (metrics,)),
             ('fig02_paired_outcomes', figure_pairs, (metrics, transitions)),
             ('fig03_historical_ranking', figure_ranking, (ranking,))]
    reports = []
    with PdfPages(HERE / 'figures_overview.pdf') as pdf:
        for name, builder, inputs in specs:
            reports.append(export(builder(*inputs), name, pdf))
    previews = []
    for name, _, _ in specs:
        with Image.open(HERE/f'{name}.png') as im:
            previews.append(ImageOps.contain(im.convert('RGB'), (1500, 900)))
    overview = Image.new('RGB', (1540, sum(im.height for im in previews)+80), 'white')
    y = 20
    for im in previews:
        overview.paste(im, ((1540-im.width)//2, y))
        y += im.height+20
    overview.save(HERE/'figures_overview.png')
    ImageOps.grayscale(overview).save(HERE/'figures_overview_grayscale.png')
    require(len(PdfReader(HERE/'figures_overview.pdf').pages) == 3, 'Overview missing pages')
    report = {'backend': 'Python / matplotlib', 'matplotlib_version': matplotlib.__version__,
              'source_mode': 'raw_refresh' if args.refresh_source else 'portable_csv',
              'seed': 2026, 'seed_replicates_per_dataset': 1, 'error_bars': None,
              'data_sha256': {p.name: sha(p) for p in sorted(DATA.glob('*.csv'))},
              'figures': reports, 'overview_pdf_pages': 3,
              'visual_review': 'See README for human visual inspection; automated bounds checks do not detect all overlaps.'}
    (HERE/'qa_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'figures': len(reports), 'direct_questions': sum(r['n'] for r in metrics),
                      'ranking_questions': 4943, 'overview_pages': 3, 'output': str(HERE)}))


if __name__ == '__main__':
    main()
