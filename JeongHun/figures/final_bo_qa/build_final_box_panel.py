#!/usr/bin/env python3
"""Build Fig 12 (final_box_panel): final BO-QA per-run Quality and TTS_0.99 boxplots.

Reconstructed builder (2026-08-05): the original figure was a throwaway heredoc
that was never committed; this script reproduces it from the committed raw
per-repetition CSV and was numerically verified against the manuscript
(3NQ9 Q med 0.945 range 0.930-0.953, TTS 1.57 ms range 1.49-1.71; 4JSZ Q med
0.985 range 0.974-0.991, TTS 1.09 ms range 0.97-1.25; n=10 per target).

Reads:
  - <JeongHun>/tables/final_bo_qa/quality_tts_raw.csv
      (ligand, repeat, quality, tts_0p99_sec, ...)

Renders a 2-panel (Quality | TTS_0.99) boxplot; box = median + IQR,
whiskers = min..max over the 10 repetitions per target.

Style (QST publication pass): no axes titles (metric on the y-axis, target on the
x-axis); (a)/(b) sub-labels; enlarged fonts. Displayed values are the committed
CSV values.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 11.5,
    "font.size": 11.5,
})

THIS_DIR = Path(__file__).resolve().parent
JH_ROOT = THIS_DIR.parent.parent
RAW_CSV = JH_ROOT / "tables/final_bo_qa/quality_tts_raw.csv"
OUT_PREFIX = THIS_DIR / "final_box_panel"

LIGANDS = ["3nq9", "4jsz"]
LIGAND_LABEL = {"3nq9": "3NQ9", "4jsz": "4JSZ"}
LIGAND_COLOR = {"3nq9": "#4C72B0", "4jsz": "#DD8452"}


def panel_label(ax, s):
    return  # (a)/(b)/(c) markers moved to the LaTeX caption per reviewer request


def load():
    q = defaultdict(list)
    t = defaultdict(list)
    for r in csv.DictReader(open(RAW_CSV, newline="")):
        q[r["ligand"]].append(float(r["quality"]))
        t[r["ligand"]].append(float(r["tts_0p99_sec"]) * 1000.0)  # sec -> ms
    return q, t


def draw_box(ax, data_by_lig, ylabel):
    data = [data_by_lig[lig] for lig in LIGANDS]
    bp = ax.boxplot(data, positions=[1, 2], widths=0.55, patch_artist=True,
                    whis=(0, 100), medianprops=dict(color="black", linewidth=1.6),
                    flierprops=dict(marker="o", markersize=4))
    for patch, lig in zip(bp["boxes"], LIGANDS):
        patch.set_facecolor(LIGAND_COLOR[lig])
        patch.set_edgecolor("black")
        patch.set_alpha(0.9)
    for whisk in bp["whiskers"]:
        whisk.set_color("#333")
    for cap in bp["caps"]:
        cap.set_color("#333")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([LIGAND_LABEL[lig] for lig in LIGANDS])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle=":")


def main():
    q, t = load()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    draw_box(axes[0], q, r"Quality $\mathcal{Q}$")
    panel_label(axes[0], "(a)")
    draw_box(axes[1], t, r"TTS$_{0.99}$ (ms)")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    for _ax in fig.axes:
        for _l in _ax.get_xticklabels() + _ax.get_yticklabels():
            _l.set_fontweight("bold")
    fig.savefig(OUT_PREFIX.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_PREFIX.with_suffix(".pdf"))
    plt.close(fig)

    # --- individual per-panel PDFs for the subfigure layout ---
    for data, ylabel, stem in ((q, r"Quality $\mathcal{Q}$", "final_quality"),
                               (t, r"TTS$_{0.99}$ (ms)", "final_tts")):
        f1, a1 = plt.subplots(figsize=(5.6, 5.3))
        draw_box(a1, data, ylabel)
        f1.tight_layout()
        for _l in a1.get_xticklabels() + a1.get_yticklabels():
            _l.set_fontweight("bold")
        f1.savefig(THIS_DIR / f"{stem}.pdf")
        plt.close(f1)

    print(f"[final_box_panel] wrote:\n  {OUT_PREFIX.with_suffix('.png')}\n  {OUT_PREFIX.with_suffix('.pdf')}"
          f"\n  + final_quality.pdf, final_tts.pdf")


if __name__ == "__main__":
    main()
