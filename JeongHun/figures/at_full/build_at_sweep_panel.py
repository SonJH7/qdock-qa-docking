#!/usr/bin/env python3
"""Build Fig 9 (at_sweep_panel): Quality and TTS_0.99 vs anneal time (log-x lines).

Reads:
  - <JeongHun>/tables/at_full/at_quality_tts_agg.csv
      (columns: ligand, annealing_time, quality_mean/min/max, tts_0p99_mean/min/max_sec)

Renders a 2-panel (Quality | TTS_0.99) line plot with both targets overlaid.
Shaded band from min to max across n_repeats=10 per (ligand, annealing_time).

Target-specific optimal anneal times marked as dashed vertical lines:
  - 3NQ9 adopted T_anneal = 800 us  (dashed blue vertical line)
  - 4JSZ TTS-minimum   T_anneal =  20 us  (dashed red  vertical line)

Style (2026-08-05, QST publication pass): axes titles removed (the metric is
already on the y-axis); (a)/(b) sub-labels added; targets stay in the legend;
paths are now script-relative so the canonical (QDock_Paper) tables are read and
the figure is written next to this script. All fonts enlarged. Data source,
computation, and displayed values (incl. the 20/800 us dashed lines) UNCHANGED.
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
JH_ROOT = THIS_DIR.parent.parent           # .../JeongHun
AT_CSV = JH_ROOT / "tables/at_full/at_quality_tts_agg.csv"
OUT_DIR = THIS_DIR
OUT_PREFIX = OUT_DIR / "at_sweep_panel"

LIGANDS = ["3nq9", "4jsz"]
LIGAND_LABEL = {"3nq9": "3NQ9", "4jsz": "4JSZ"}
LIGAND_COLOR = {"3nq9": "#1f77b4", "4jsz": "#d62728"}
LIGAND_MARKER = {"3nq9": "o", "4jsz": "s"}

# Optional visual markers at target-specific optima
OPTIMAL_ANNEAL_US = {"3nq9": 800.0, "4jsz": 20.0}


def panel_label(ax, s):
    return  # (a)/(b)/(c) markers moved to the LaTeX caption per reviewer request


def to_float(s, default=math.nan) -> float:
    try:
        s = str(s).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def load_series(rows: list[dict], ligand: str, metric_prefix: str) -> tuple[list[float], list[float], list[float], list[float]]:
    ats = []
    means = []
    lows = []
    highs = []
    sel = [r for r in rows if r["ligand"] == ligand]
    sel.sort(key=lambda r: to_float(r["annealing_time"]))
    for r in sel:
        at = to_float(r["annealing_time"])
        m = to_float(r[f"{metric_prefix}_mean" + ("_sec" if metric_prefix.startswith("tts") else "")])
        lo = to_float(r[f"{metric_prefix}_min" + ("_sec" if metric_prefix.startswith("tts") else "")])
        hi = to_float(r[f"{metric_prefix}_max" + ("_sec" if metric_prefix.startswith("tts") else "")])
        if all(math.isfinite(v) for v in (at, m, lo, hi)):
            ats.append(at)
            means.append(m)
            lows.append(lo)
            highs.append(hi)
    return ats, means, lows, highs


def draw_quality_panel(ax, rows) -> None:
    for lig in LIGANDS:
        ats, means, lows, highs = load_series(rows, lig, "quality")
        c = LIGAND_COLOR[lig]
        ax.plot(ats, means, marker=LIGAND_MARKER[lig], color=c,
                label=LIGAND_LABEL[lig], linewidth=1.7, markersize=6, zorder=3)
        ax.fill_between(ats, lows, highs, color=c, alpha=0.18, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Anneal time $T_{\mathrm{anneal}}$ ($\mu$s)")
    ax.set_ylabel(r"Quality $\mathcal{Q}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right")

    for lig, at_opt in OPTIMAL_ANNEAL_US.items():
        ax.axvline(at_opt, color=LIGAND_COLOR[lig], linestyle="--",
                   linewidth=1.1, alpha=0.55, zorder=1)


def draw_tts_panel(ax, rows) -> None:
    for lig in LIGANDS:
        ats, means_sec, lows_sec, highs_sec = load_series(rows, lig, "tts_0p99")
        means_ms = [v * 1000.0 for v in means_sec]
        lows_ms = [v * 1000.0 for v in lows_sec]
        highs_ms = [v * 1000.0 for v in highs_sec]
        c = LIGAND_COLOR[lig]
        ax.plot(ats, means_ms, marker=LIGAND_MARKER[lig], color=c,
                label=LIGAND_LABEL[lig], linewidth=1.7, markersize=6, zorder=3)
        ax.fill_between(ats, lows_ms, highs_ms, color=c, alpha=0.18, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Anneal time $T_{\mathrm{anneal}}$ ($\mu$s)")
    ax.set_ylabel(r"TTS$_{0.99}$ (ms)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left")

    for lig, at_opt in OPTIMAL_ANNEAL_US.items():
        ax.axvline(at_opt, color=LIGAND_COLOR[lig], linestyle="--",
                   linewidth=1.1, alpha=0.55, zorder=1)


def main() -> None:
    with AT_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    draw_quality_panel(axes[0], rows)
    panel_label(axes[0], "(a)")
    draw_tts_panel(axes[1], rows)
    panel_label(axes[1], "(b)")
    fig.tight_layout()

    for _ax in fig.axes:
        for _l in _ax.get_xticklabels() + _ax.get_yticklabels():
            _l.set_fontweight("bold")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_PREFIX.with_suffix(".png")
    pdf = OUT_PREFIX.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    # --- individual per-panel PDFs for the subfigure layout ---
    for draw, stem in ((draw_quality_panel, "at_quality"), (draw_tts_panel, "at_tts")):
        f1, a1 = plt.subplots(figsize=(6.6, 5.2))
        draw(a1, rows)
        f1.tight_layout()
        for _l in a1.get_xticklabels() + a1.get_yticklabels():
            _l.set_fontweight("bold")
        f1.savefig(OUT_DIR / f"{stem}.pdf")
        plt.close(f1)

    print(f"[at_sweep_panel] wrote:\n  {png}\n  {pdf}\n  + at_quality.pdf, at_tts.pdf")


if __name__ == "__main__":
    main()
