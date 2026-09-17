"""Build the per-run final-Quality strip plot (Figure 7 in new.tex, fig:bo_strip).

Reads the 20-run comparison (five runs per policy and target) from
`tables/bo_vs_random/final_best_per_run.csv` and the statistics from
`tables/bo_vs_random/stat_tests.csv`, and annotates the two-sided Mann--Whitney
p-value (Holm-adjusted) and Cliff's delta per target.

Outputs: figures/bo_vs_random/bo_random_strip.{png,pdf}
(Committed builder added 2026-07-31; the prior figure had no committed script.)

Style (2026-08-05, QST publication pass): axes titles removed; (a)/(b)
sub-labels added; the target name (was in the title) moved into the legend
title, per-group n moved from the title into the x-tick labels; all fonts
enlarged.
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 11.5,
    "legend.title_fontsize": 12.5,
    "font.size": 11.5,
})

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
FINAL = os.path.join(ROOT, "tables/bo_vs_random/final_best_per_run.csv")
STATS = os.path.join(ROOT, "tables/bo_vs_random/stat_tests.csv")
COL = {"bo": "#1f4e8a", "random": "#c92a2a"}


def panel_label(ax, s):
    return  # (a)/(b)/(c) markers moved to the LaTeX caption per reviewer request


def load():
    runs = {}
    for r in csv.DictReader(open(FINAL)):
        runs.setdefault((r["ligand"], r["method"]), []).append(float(r["final_best_so_far"]))
    stat = {}
    for r in csv.DictReader(open(STATS)):
        if r["metric"] == "final_best":
            stat[r["ligand"]] = (float(r["mannwhitney_p_two_sided"]), float(r["cliffs_delta"]),
                                 float(r["holm_p"]), int(r["bo_n"]), int(r["random_n"]))
    return runs, stat


def panel(ax, runs, stat, lig, target_label, legend_in_title=True):
    rng = np.random.default_rng(42)
    for xi, meth in enumerate(("bo", "random")):
        v = np.array(runs[(lig, meth)])
        x = xi + (rng.random(len(v)) - 0.5) * 0.22
        ax.scatter(x, v, s=70, color=COL[meth], edgecolor="#222", linewidth=0.6, zorder=3, alpha=0.9)
        ax.hlines(v.mean(), xi - 0.28, xi + 0.28, color=COL[meth], lw=3, zorder=4)
    p, d, holm, bn, rn = stat[lig]
    ax.text(0.97, 0.03,
            f"Mann-Whitney $p$={p:.2f} (Holm {holm:.2f})\nCliff's $\\delta$={d:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
            bbox=dict(boxstyle="round", fc="white", ec="#aaa"))
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 1])
    # Per-group n moved out of the (removed) title into the x-tick labels.
    ax.set_xticklabels([f"BO\n($n$={bn})", f"Random\n($n$={rn})"])
    ax.set_ylabel("Final best-so-far Quality (run end)")
    # Target identity (was the axes title) carried by the legend title.
    handles = [Line2D([0], [0], color="#444", lw=3, label="Group mean")]
    ax.legend(handles=handles, loc="upper left",
              title=(target_label if legend_in_title else None),
              frameon=True, framealpha=0.92, edgecolor="#aaa")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    for s in ax.spines.values():
        s.set_color("#444")


def main():
    runs, stat = load()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    panel(axes[0], runs, stat, "3nq9", "3NQ9")
    panel_label(axes[0], "(a)")
    panel(axes[1], runs, stat, "4jsz", "4JSZ")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    for _ax in fig.axes:
        for _l in _ax.get_xticklabels() + _ax.get_yticklabels():
            _l.set_fontweight("bold")
    fig.savefig(os.path.join(THIS, "bo_random_strip.png"), dpi=300)
    fig.savefig(os.path.join(THIS, "bo_random_strip.pdf"))
    plt.close(fig)

    # --- individual per-panel PDFs for the subfigure layout ---
    for lig, tgt, stem in (("3nq9", "3NQ9", "bo_strip_3nq9"), ("4jsz", "4JSZ", "bo_strip_4jsz")):
        f1, a1 = plt.subplots(figsize=(6.0, 5.4))
        panel(a1, runs, stat, lig, tgt, legend_in_title=False)
        f1.tight_layout()
        for _l in a1.get_xticklabels() + a1.get_yticklabels():
            _l.set_fontweight("bold")
        f1.savefig(os.path.join(THIS, f"{stem}.pdf"))
        plt.close(f1)

    print("[OK] bo_random_strip.{png,pdf} + bo_strip_3nq9.pdf, bo_strip_4jsz.pdf")


if __name__ == "__main__":
    main()
