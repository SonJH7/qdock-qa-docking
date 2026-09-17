#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 11.0,
    "legend.title_fontsize": 12.5,
    "font.size": 11.5,
})

THIS_DIR = Path(__file__).resolve().parent
JH_ROOT = THIS_DIR.parent.parent
AS_JJIN_CSV = JH_ROOT / "tables/as_suite/as_jjin_quality_tts_agg.csv"
OUT_PREFIX = THIS_DIR / "as_schedule_compare"

LIGANDS = ["3nq9", "4jsz"]
LIGAND_LABEL = {"3nq9": "3NQ9", "4jsz": "4JSZ"}
MODE = "anneal_schedule"
# Six parameterized variants, alphabetical (as in the committed figure).
VARIANTS = ["bangbang_monotone", "cubic_smoothstep", "fourier",
            "linear", "lowpass_linear", "pause_quench"]
BAR_COLOR = "#4C72B0"


def panel_label(ax, s):
    return  # (a)/(b)/(c) markers moved to the LaTeX caption per reviewer request


def to_float(s, default=math.nan) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return default


def find(rows, lig, variant):
    for r in rows:
        if r["ligand"] == lig and r["mode"] == MODE and r["variant"] == variant:
            return r
    return None


def build_panel(ax, rows, lig, legend_in_title=True):
    means, lows, highs = [], [], []
    for v in VARIANTS:
        r = find(rows, lig, v)
        if r is None:
            raise RuntimeError(f"missing row: {lig} {v}")
        q = to_float(r["quality_mean"])
        means.append(q)
        lows.append(max(0.0, q - to_float(r["quality_min"])))
        highs.append(max(0.0, to_float(r["quality_max"]) - q))
    x = list(range(len(VARIANTS)))
    ax.bar(x, means, color=BAR_COLOR, alpha=0.9, edgecolor="black",
           linewidth=0.6, width=0.68, zorder=2)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor="#202020",
                elinewidth=1.1, capsize=5, capthick=1.1, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, rotation=28, ha="right")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Quality")
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.set_axisbelow(True)
    handle = Patch(facecolor=BAR_COLOR, edgecolor="black",
                   label="Mean Quality (10 reps; min/max)")
    ax.legend(handles=[handle], loc="upper right",
              title=(LIGAND_LABEL[lig] if legend_in_title else None),
              frameon=True, framealpha=0.92, edgecolor="#aaa")


def main():
    with AS_JJIN_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    for ax, lig in zip(axes, LIGANDS):
        build_panel(ax, rows, lig)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    for _ax in fig.axes:
        for _l in _ax.get_xticklabels() + _ax.get_yticklabels():
            _l.set_fontweight("bold")
    fig.savefig(OUT_PREFIX.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_PREFIX.with_suffix(".pdf"))
    plt.close(fig)

    # --- individual per-panel PDFs for the subfigure layout ---
    for lig, stem in (("3nq9", "as_compare_3nq9"), ("4jsz", "as_compare_4jsz")):
        f1, a1 = plt.subplots(figsize=(6.4, 5.6))
        build_panel(a1, rows, lig, legend_in_title=False)
        f1.tight_layout()
        for _l in a1.get_xticklabels() + a1.get_yticklabels():
            _l.set_fontweight("bold")
        # bbox_inches="tight" so the leftmost rotated x-label is not clipped
        f1.savefig(THIS_DIR / f"{stem}.pdf", bbox_inches="tight")
        plt.close(f1)

    print(f"[as_schedule_compare] wrote:\n  {OUT_PREFIX.with_suffix('.png')}\n  {OUT_PREFIX.with_suffix('.pdf')}"
          f"\n  + as_compare_3nq9.pdf, as_compare_4jsz.pdf")


if __name__ == "__main__":
    main()
