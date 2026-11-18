"""Render the manuscript-style Figure 2 and Figure S1 at 300 dpi."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("phosphate_only", "phosphate_creatinine", "six_variable")
SOURCES = {"mimic": "MIMIC-IV", "eicu": "eICU-CRD", "sicdb": "SICdb"}
LABELS = dict(zip(MODELS, ("Phosphate", "Phosphate + creatinine", "Six variables")))
COLORS = dict(zip(MODELS, ("#7B8794", "#238B8B", "#24466B")))
LINES = dict(zip(MODELS, ("--", "-.", "-")))
RULE_COLORS = {2.5: "#8A4D48", 3.: "#98713D"}
RULE_MARKS = {2.5: "D", 3.: "s"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                     "svg.fonttype": "none", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 10, "axes.titlesize": 11, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "savefig.facecolor": "white"})


def save(fig, out, name, qa):
    fig.canvas.draw()
    if qa:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=out / "qa" / f"{name}_alignment.json",
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def figure2(root, results, out, qa):
    frozen_results = root / "results"
    performance = pd.read_csv(frozen_results / "performance.csv").query("metric=='auc'")
    grouped = pd.read_csv(frozen_results / "calibration_groups.csv")
    loess = pd.read_csv(results / "B10_loess.csv")
    roc_grid = pd.read_csv(results / "B10_ROC_display.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    fig.subplots_adjust(left=.085, right=.974, top=.915, bottom=.145, hspace=.44, wspace=.265)
    for i, (source, title) in enumerate(SOURCES.items()):
        data = root / "private" / f"{source}_predictions.parquet"
        predictions = pd.read_parquet(data) if data.is_file() else None
        top, bottom = axes[:, i]
        top.plot([0, 1], [0, 1], ls=":", c="#ADB4BB", lw=1)
        bottom.plot([0, .55], [0, .55], ls=":", c="#ADB4BB", lw=1)
        for model in MODELS:
            if predictions is not None:
                x, y, _ = roc_curve(predictions.event, predictions[model])
            else:
                grid = roc_grid.query("source == @source and model == @model")
                x, y = grid.false_positive_fraction, grid.sensitivity
            auc = performance.query("source == @source and model == @model").estimate.item()
            top.plot(x, y, c=COLORS[model], ls=LINES[model], lw=1.6, label=f"{LABELS[model]}  {auc:.3f}")
            points = grouped.query("source == @source and model == @model")
            curve = loess.query("source == @source and model == @model")
            bottom.fill_between(curve.predicted, curve.ci_low, curve.ci_high, color=COLORS[model],
                                alpha=.16, edgecolor="none")
            bottom.scatter(points.predicted, points.observed, color=COLORS[model], alpha=.40,
                           s=17, edgecolor="none")
            bottom.plot(curve.predicted, curve.estimate, c=COLORS[model], ls=LINES[model], lw=1.8,
                        label=LABELS[model])
        top.legend(loc="lower right", fontsize=7.7, frameon=False)
        top.set(xlabel="False-positive fraction", ylabel="Sensitivity" if i == 0 else "",
                xlim=(0, 1), ylim=(0, 1), title=title)
        bottom.set(xlabel="Predicted probability", ylabel="Observed proportion" if i == 0 else "",
                   xlim=(0, .55), ylim=(0, .55), xticks=np.arange(0, .51, .1), yticks=np.arange(0, .51, .1))
        top.text(.038, 1.068, chr(65+i), transform=top.transAxes, fontweight="bold", fontsize=12)
        bottom.text(.038, 1.068, chr(68+i), transform=bottom.transAxes, fontweight="bold", fontsize=12)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .025), ncol=3, frameon=False)
    save(fig, out, "Figure_2_discrimination_calibration", qa)


def figure_s1(root, results, out, qa):
    decisions = pd.read_csv(root / "results" / "decision_curves.csv")
    rules = pd.read_csv(results / "E2_net_benefit.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.subplots_adjust(left=.093, right=.977, top=.868, bottom=.27, wspace=.265)
    for i, (source, title) in enumerate(SOURCES.items()):
        ax = axes[i]
        for model in MODELS:
            curve = decisions.query("source == @source and model == @model")
            ax.plot(curve.threshold, curve.net_benefit, c=COLORS[model], ls=LINES[model], lw=1.75,
                    label=LABELS[model])
        # Crop the default strategy to the visible y-range before vector export.
        # The curve is monotone, so its one crossing is obtained by interpolation.
        test_all = curve.monitor_all.to_numpy()
        threshold = curve.threshold.to_numpy()
        visible = test_all >= -.04
        edge = np.interp(-.04, test_all[::-1], threshold[::-1])
        ax.plot(np.r_[threshold[visible], edge], np.r_[test_all[visible], -.04],
                c="#B68158", ls="--", lw=1.3, label="Test all")
        ax.axhline(0, c="#777777", ls=":", lw=1.3, label="Test none")
        for cutoff in RULE_COLORS:
            subset = rules.loc[rules.source.eq(source) & rules.baseline_phosphate_lt_mg_dl.eq(cutoff)]
            ax.scatter(subset.threshold, subset.estimate, s=39, marker=RULE_MARKS[cutoff],
                       edgecolor=RULE_COLORS[cutoff], linewidth=1.25, facecolor="white",
                       zorder=4, clip_on=False, label=f"Phosphate <{cutoff:.1f} mg/dL")
        ax.set(title=title, xlabel="Risk threshold", ylabel="Net benefit" if i == 0 else "",
               xlim=(.047, .303), ylim=(-.04, .16), xticks=[.05, .10, .15, .20, .25, .30])
        ax.text(.012, 1.038, chr(65+i), transform=ax.transAxes, fontweight="bold", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .023), ncol=4, frameon=False,
               fontsize=8.7, handlelength=2.4, columnspacing=1.35)
    save(fig, out, "Figure_S1_exploratory_decision_curves", qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--figure-audit-scripts", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    revision = root / "revision_20261116"
    results = revision / "results" / "stage2"
    output = revision / "figures"
    output.mkdir(parents=True, exist_ok=True)
    if args.figure_audit_scripts:
        sys.path.insert(0, str(args.figure_audit_scripts.resolve()))
        (output / "qa").mkdir(exist_ok=True)
    figure2(root, results, output, args.figure_audit_scripts)
    figure_s1(root, results, output, args.figure_audit_scripts)

