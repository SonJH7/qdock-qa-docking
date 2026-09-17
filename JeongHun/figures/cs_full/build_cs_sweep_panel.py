#!/usr/bin/env python3
"""Build Fig 8 (cs_sweep_panel): Quality / TTS_0.99 / chain-break fraction vs chain strength.

Reconstructed builder (2026-08-05): the original figure was produced by a
throwaway heredoc that was never committed; this script reproduces it from the
committed aggregate CSV and was numerically verified against the manuscript
(3NQ9 peak cs=1.0 Q=0.879/max 0.889, chain-break 0.890; 4JSZ peak cs=0.75
Q=0.947/max 0.990, chain-break 0.632).

Reads:
  - <JeongHun>/tables/cs_full/cs_quality_tts_agg.csv
      (ligand, chain_strength, quality_{mean,min,max},
       tts_0p99_{mean,min,max}_sec, chain_break_fraction_{mean,min,max})

Renders a 3-panel (Quality | TTS_0.99 | chain-break fraction) line plot with both
targets overlaid; shaded band = min..max over n_repeats per (ligand, cs).

Style (QST publication pass): no axes titles (metric on the y-axis); (a)/(b)/(c)
sub-labels; targets in the legend; enlarged fonts; marker/colour convention
matched to Fig 9 (at_sweep). Displayed values are the committed CSV values.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 11.5,
    "font.size": 11.5,
})

THIS_DIR = Path(__file__).resolve().parent
JH_ROOT = THIS_DIR.parent.parent
CS_CSV = JH_ROOT / "tables/cs_full/cs_quality_tts_agg.csv"
OUT_PREFIX = THIS_DIR / "cs_sweep_panel"

LIGANDS = ["3nq9", "4jsz"]
LIGAND_LABEL = {"3nq9": "3NQ9", "4jsz": "4JSZ"}
LIGAND_COLOR = {"3nq9": "#1f77b4", "4jsz": "#d62728"}
LIGAND_MARKER = {"3nq9": "o", "4jsz": "s"}


def panel_label(ax, s):
    return  # (a)/(b)/(c) markers moved to the LaTeX caption per reviewer request


def to_float(s, default=math.nan) -> float:
    try:
        s = str(s).strip()
        if s == "" or s.lower() in {"nan", "none", "inf"}:
            return default if s.lower() != "inf" else math.inf
        return float(s)
    except Exception:
        return default


def series(rows, ligand, mean_key, lo_key, hi_key, scale=1.0):
    xs, ms, los, his = [], [], [], []
    sel = [r for r in rows if r["ligand"] == ligand]
    sel.sort(key=lambda r: to_float(r["chain_strength"]))
    for r in sel:
        cs = to_float(r["chain_strength"])
        m = to_float(r[mean_key]) * scale
        lo = to_float(r[lo_key]) * scale
        hi = to_float(r[hi_key]) * scale
        if all(math.isfinite(v) for v in (cs, m, lo, hi)):
            xs.append(cs); ms.append(m); los.append(lo); his.append(hi)
    return xs, ms, los, his


def draw_panel(ax, rows, mean_key, lo_key, hi_key, ylabel, scale=1.0,
               logy=False, legend_loc="best"):
    for lig in LIGANDS:
        xs, ms, los, his = series(rows, lig, mean_key, lo_key, hi_key, scale)
        c = LIGAND_COLOR[lig]
        ax.plot(xs, ms, marker=LIGAND_MARKER[lig], color=c, label=LIGAND_LABEL[lig],
                linewidth=1.7, markersize=6, zorder=3)
        ax.fill_between(xs, los, his, color=c, alpha=0.18, zorder=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"Chain strength $cs$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc=legend_loc)


def main():
    with CS_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    draw_panel(axes[0], rows, "quality_mean", "quality_min", "quality_max",
               r"Quality $\mathcal{Q}$", legend_loc="upper right")
    draw_panel(axes[1], rows, "tts_0p99_mean_sec", "tts_0p99_min_sec", "tts_0p99_max_sec",
               r"TTS$_{0.99}$ (ms)", scale=1000.0, logy=True, legend_loc="lower right")
    draw_panel(axes[2], rows, "chain_break_fraction_mean", "chain_break_fraction_min",
               "chain_break_fraction_max", "Chain-break fraction", legend_loc="upper right")
    for ax, tag in zip(axes, ["(a)", "(b)", "(c)"]):
        panel_label(ax, tag)
    fig.tight_layout()

    for _ax in fig.axes:
        for _l in _ax.get_xticklabels() + _ax.get_yticklabels():
            _l.set_fontweight("bold")
    fig.savefig(OUT_PREFIX.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_PREFIX.with_suffix(".pdf"))
    plt.close(fig)

    # --- individual per-panel PDFs for the subfigure layout ---
    panels = [
        (("quality_mean", "quality_min", "quality_max"),
         r"Quality $\mathcal{Q}$", 1.0, False, "upper right", "cs_quality"),
        (("tts_0p99_mean_sec", "tts_0p99_min_sec", "tts_0p99_max_sec"),
         r"TTS$_{0.99}$ (ms)", 1000.0, True, "lower right", "cs_tts"),
        (("chain_break_fraction_mean", "chain_break_fraction_min", "chain_break_fraction_max"),
         "Chain-break fraction", 1.0, False, "upper right", "cs_chainbreak"),
    ]
    for (mk, lk, hk), ylabel, scale, logy, loc, stem in panels:
        f1, a1 = plt.subplots(figsize=(6.2, 5.3))
        draw_panel(a1, rows, mk, lk, hk, ylabel, scale=scale, logy=logy, legend_loc=loc)
        f1.tight_layout()
        for _l in a1.get_xticklabels() + a1.get_yticklabels():
            _l.set_fontweight("bold")
        f1.savefig(THIS_DIR / f"{stem}.pdf")
        plt.close(f1)

    print(f"[cs_sweep_panel] wrote:\n  {OUT_PREFIX.with_suffix('.png')}\n  {OUT_PREFIX.with_suffix('.pdf')}"
          f"\n  + cs_quality.pdf, cs_tts.pdf, cs_chainbreak.pdf")


if __name__ == "__main__":
    main()
