#!/usr/bin/env python3
"""Build Fig 5 (penalty landscape): combined Quality heatmaps + Top-10 error bars.

Consolidated, self-contained builder for the three canonical Fig 5 image files
that new.tex includes from figures/penalty_pub/:
  - quality_heatmap_combined.pdf              (2-panel best-of-5 Quality heatmap)
  - penalty_quality_top10_errorbar_3nq9_5run.png
  - penalty_quality_top10_errorbar_4jsz_5run.png

Data source (unchanged): the 5 QPU repetition case-metrics CSVs
  penalty_sweep_qpu_v113/results/penalty_sweep_case_metrics_*.csv
The heatmap value + star per cell is the best-of-5 (max Quality over the 5 reps,
with validity/mRMSD from that same max-quality rep) exactly as in the committed
make_pub_heatmaps_v113 (5rep_union) pipeline; the Top-10 error bars rank cells by
5-run MEAN Quality (mean +/- std) exactly as in analyze_penalty_stats_v113. Both
were numerically verified (2026-08-05) to reproduce the committed values:
3NQ9 star (6.5,2.0) Q=0.869; 4JSZ star (6.0,7.0) Q=0.976.

Style (2026-08-05, QST publication pass): axes titles + suptitle removed;
(a)/(b) sub-labels on the heatmap panels and (c)/(d) on the two error bars; the
target name (was the axes title) carried by the heatmap star box / the error-bar
legend title; all fonts enlarged. Displayed values are UNCHANGED.
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 11.5,
    "legend.title_fontsize": 12.5,
    "font.size": 11.5,
})

# --- Source data (only location of the raw rep CSVs) ---
REPO_ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[3]))
).expanduser().resolve()
SRC_RESULTS = REPO_ROOT / "penalty_sweep_qpu_v113/results"
THIS_DIR = Path(__file__).resolve().parent
OUT_HEATMAP = THIS_DIR / "quality_heatmap_combined"
OUT_EB = {lig: THIS_DIR / f"penalty_quality_top10_errorbar_{lig}_5run.png"
          for lig in ("3nq9", "4jsz")}

LIGAND_LABEL = {"3nq9": "3NQ9", "4jsz": "4JSZ"}
TOP_K = 10


def panel_label(ax, s, x=-0.12, y=1.05):
    return  # (a)/(b)/(c) markers moved to the LaTeX caption per reviewer request


# ---------------- data loaders (identical logic to committed pipeline) --------
def load_all_reps():
    """best-of-N per (ligand,Kdist,Kmono): keep the max-quality rep's row."""
    csvs = sorted(SRC_RESULTS.glob("penalty_sweep_case_metrics_*.csv"))
    best = {}
    for csv_path in csvs:
        with csv_path.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                row = {
                    "ligand": r["ligand"],
                    "Kdist": float(r["Kdist"]),
                    "Kmono": float(r["Kmono"]),
                    "quality": float(r["quality"]),
                    "validity_rate": float(r["validity_rate"]),
                    "mRMSD": float(r["mRMSD"]) if r.get("mRMSD") not in ("", None) else float("inf"),
                }
                key = (row["ligand"], row["Kdist"], row["Kmono"])
                cur = best.get(key)
                if cur is None or (-row["quality"], -row["validity_rate"], row["mRMSD"]) \
                        < (-cur["quality"], -cur["validity_rate"], cur["mRMSD"]):
                    best[key] = row
    return list(best.values())


def choose_best(rows):
    return sorted(rows, key=lambda r: (-r["quality"], -r["validity_rate"], r["mRMSD"]))[0]


def build_matrix(rows):
    kdist_vals = sorted({r["Kdist"] for r in rows})
    kmono_vals = sorted({r["Kmono"] for r in rows})
    mat = np.full((len(kdist_vals), len(kmono_vals)), np.nan)
    ki = {v: i for i, v in enumerate(kdist_vals)}
    mi = {v: i for i, v in enumerate(kmono_vals)}
    for r in rows:
        mat[ki[r["Kdist"]], mi[r["Kmono"]]] = r["quality"]
    return mat, kdist_vals, kmono_vals, ki, mi


def load_mean_ranked(top_k):
    """5-run MEAN Quality per complete cell, ranked; returns {lig: [(kd,km,mean,std)]}."""
    files = sorted(glob.glob(str(SRC_RESULTS / "penalty_sweep_case_metrics_*.csv")))
    run_tags = [Path(f).name.replace("penalty_sweep_case_metrics_", "").replace(".csv", "")
                for f in files]
    per = defaultdict(dict)
    for f, rt in zip(files, run_tags):
        for r in csv.DictReader(open(f, newline="")):
            if (r.get("status") or "").strip().lower() != "ok":
                continue
            lig = (r.get("ligand") or "").strip().lower()
            if lig not in ("3nq9", "4jsz"):
                continue
            per[(lig, float(r["Kdist"]), float(r["Kmono"]))][rt] = float(r["quality"])
    out = {}
    for lig in ("3nq9", "4jsz"):
        rows = []
        for key, m in per.items():
            if key[0] != lig:
                continue
            if all(rt in m for rt in run_tags):
                arr = np.array([m[rt] for rt in run_tags], dtype=float)
                rows.append((key[1], key[2], float(arr.mean()), float(np.std(arr, ddof=0))))
        rows.sort(key=lambda t: (t[2], -t[3], -t[0], -t[1]), reverse=True)
        out[lig] = rows[:top_k]
    return out


# ---------------- drawing ------------------------------------------------------
def stylize_axes(ax, kdist_vals, kmono_vals):
    ax.set_xlabel(r"$K_{\mathrm{mono}}$")
    ax.set_ylabel(r"$K_{\mathrm{dist}}$")
    ax.set_xticks(range(len(kmono_vals)))
    ax.set_yticks(range(len(kdist_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in kmono_vals], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:.1f}" for v in kdist_vals])
    ax.set_xticks(np.arange(-0.5, len(kmono_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(kdist_vals), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_star(ax, best, ki, mi, target, show_target=True):
    i = ki[best["Kdist"]]
    j = mi[best["Kmono"]]
    ax.scatter([j], [i], marker="*", s=300, facecolor="#4dd0e1",
               edgecolor="black", linewidth=1.0, zorder=6)
    # Target identity (was the axes title) folded into the best-cell data box;
    # omit it in the split layout where the subcaption already names the target.
    prefix = f"{target}   " if show_target else ""
    info = (f"{prefix}best-of-5: "
            f"($K_{{\\mathrm{{dist}}}}$={best['Kdist']:.1f}, $K_{{\\mathrm{{mono}}}}$={best['Kmono']:.1f})\n"
            f"Quality = {best['quality']:.3f},  Validity = {best['validity_rate']:.3f}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=11.0, bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, pad=3.0))


def build_heatmap(rows):
    rows_by_lig = defaultdict(list)
    for r in rows:
        rows_by_lig[r["ligand"]].append(r)
    ligands = sorted(rows_by_lig.keys())
    vmax = max(r["quality"] for r in rows)

    fig, axes = plt.subplots(1, len(ligands), figsize=(17.0, 7.4),
                             constrained_layout=True)
    if len(ligands) == 1:
        axes = [axes]
    im = None
    for ax, lig, tag in zip(axes, ligands, ["(a)", "(b)"]):
        lr = rows_by_lig[lig]
        mat, kdv, kmv, ki, mi = build_matrix(lr)
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
        stylize_axes(ax, kdv, kmv)
        draw_star(ax, choose_best(lr), ki, mi, LIGAND_LABEL[lig])
        panel_label(ax, tag)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"Quality $\mathcal{Q}$")
    for _ax in fig.axes:
        for _l in _ax.get_xticklabels() + _ax.get_yticklabels():
            _l.set_fontweight("bold")
    fig.savefig(OUT_HEATMAP.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_HEATMAP.with_suffix(".pdf"))
    plt.close(fig)

    # --- individual per-target heatmaps for the subfigure layout ---
    for lig in ligands:
        lr = rows_by_lig[lig]
        mat, kdv, kmv, ki, mi = build_matrix(lr)
        f1, a1 = plt.subplots(figsize=(8.8, 7.2), constrained_layout=True)
        im1 = a1.imshow(mat, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
        stylize_axes(a1, kdv, kmv)
        draw_star(a1, choose_best(lr), ki, mi, LIGAND_LABEL[lig], show_target=False)
        cb = f1.colorbar(im1, ax=a1, fraction=0.046, pad=0.03)
        cb.set_label(r"Quality $\mathcal{Q}$")
        for _l in a1.get_xticklabels() + a1.get_yticklabels():
            _l.set_fontweight("bold")
        f1.savefig(THIS_DIR / f"heatmap_{lig}.pdf")
        plt.close(f1)


def build_errorbar(lig, ranked, tag):
    labels = [f"({kd:.1f},{km:.1f})" for kd, km, _, _ in ranked]
    means = np.array([m for _, _, m, _ in ranked])
    stds = np.array([s for _, _, _, s in ranked])
    x = np.arange(len(ranked))
    fig, ax = plt.subplots(figsize=(max(9.5, 0.95 * len(ranked)), 5.2))
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=4, lw=1.5,
                markersize=7, color="#2C7FB8", label="5-run mean $\\pm$ std")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13.0)
    ax.set_ylabel("Quality (mean $\\pm$ std)")
    ax.set_xlabel(r"Penalty cell $(K_{\mathrm{dist}}, K_{\mathrm{mono}})$")
    ax.grid(axis="y", alpha=0.25)
    # The subcaption names the target in the split layout, so the legend needs
    # no target title here.
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#aaa")
    panel_label(ax, tag, x=-0.10, y=1.04)
    fig.tight_layout()
    for _l in ax.get_xticklabels() + ax.get_yticklabels():
        _l.set_fontweight("bold")
    fig.savefig(OUT_EB[lig], dpi=300)
    fig.savefig(OUT_EB[lig].with_suffix(".pdf"))  # vector version for the paper
    plt.close(fig)


def main():
    rows = load_all_reps()
    if not rows:
        raise SystemExit(f"No rep CSVs found under {SRC_RESULTS}")
    build_heatmap(rows)

    ranked = load_mean_ranked(TOP_K)
    build_errorbar("3nq9", ranked["3nq9"], "(c)")
    build_errorbar("4jsz", ranked["4jsz"], "(d)")

    print("[penalty_pub] wrote:")
    print(f"  {OUT_HEATMAP.with_suffix('.pdf')}")
    for lig in ("3nq9", "4jsz"):
        print(f"  {OUT_EB[lig]}")


if __name__ == "__main__":
    main()
