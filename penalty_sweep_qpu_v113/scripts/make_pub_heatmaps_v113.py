#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
RESULT_DIR = ROOT / "penalty_sweep_qpu_v113" / "results"
FIG_DIR = ROOT / "penalty_sweep_qpu_v113" / "figures"


def find_latest_run_tag():
    files = sorted(RESULT_DIR.glob("penalty_sweep_case_metrics_*.csv"))
    if not files:
        return None
    return files[-1].stem.replace("penalty_sweep_case_metrics_", "")


def load_rows(case_csv: Path):
    rows = []
    with case_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            rows.append(
                {
                    "ligand": r["ligand"],
                    "Kdist": float(r["Kdist"]),
                    "Kmono": float(r["Kmono"]),
                    "quality": float(r["quality"]),
                    "validity_rate": float(r["validity_rate"]),
                    "mRMSD": float(r["mRMSD"]) if r.get("mRMSD") not in ("", None) else float("inf"),
                }
            )
    return rows


def load_all_reps():
    """Aggregate ALL penalty_sweep_case_metrics_*.csv reps into best-of-N per cell.

    For each (ligand, Kdist, Kmono) cell, keep the row from the rep whose
    quality is maximal. validity_rate and mRMSD come from that same max-quality
    rep (they are NOT averaged), so the star position and the colormap value
    at that cell remain consistent.

    Returns
    -------
    rows : list of dict
        One dict per (ligand, Kdist, Kmono) cell with keys
        {ligand, Kdist, Kmono, quality, validity_rate, mRMSD}.
    csvs : list of Path
        The CSV files that were aggregated (for logging/audit).
    """
    csvs = sorted(RESULT_DIR.glob("penalty_sweep_case_metrics_*.csv"))
    if not csvs:
        return [], []

    print(f"[load_all_reps] Aggregating best-of-{len(csvs)} across CSVs:")
    for c in csvs:
        print(f"  - {c.name}")

    best_per_cell = {}  # (ligand, Kdist, Kmono) -> row dict at that cell's max quality
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
                cur = best_per_cell.get(key)
                if cur is None:
                    best_per_cell[key] = row
                    continue
                # Same tiebreak order as choose_best: higher quality wins;
                # on ties prefer higher validity, then lower mRMSD.
                cur_key = (-cur["quality"], -cur["validity_rate"], cur["mRMSD"])
                new_key = (-row["quality"], -row["validity_rate"], row["mRMSD"])
                if new_key < cur_key:
                    best_per_cell[key] = row

    return list(best_per_cell.values()), csvs


def choose_best(rows):
    return sorted(
        rows,
        key=lambda r: (-r["quality"], -r["validity_rate"], r["mRMSD"]),
    )[0]


def build_matrix(rows, metric):
    kdist_vals = sorted({r["Kdist"] for r in rows})
    kmono_vals = sorted({r["Kmono"] for r in rows})
    mat = np.full((len(kdist_vals), len(kmono_vals)), np.nan)
    kdist_idx = {v: i for i, v in enumerate(kdist_vals)}
    kmono_idx = {v: i for i, v in enumerate(kmono_vals)}
    for r in rows:
        i = kdist_idx[r["Kdist"]]
        j = kmono_idx[r["Kmono"]]
        mat[i, j] = r[metric]
    return mat, kdist_vals, kmono_vals, kdist_idx, kmono_idx


def stylize_axes(ax, kdist_vals, kmono_vals):
    ax.set_xlabel("Kmono")
    ax.set_ylabel("Kdist")
    ax.set_xticks(range(len(kmono_vals)))
    ax.set_yticks(range(len(kdist_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in kmono_vals], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"{v:.1f}" for v in kdist_vals], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(kmono_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(kdist_vals), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_best_marker(ax, best, kdist_idx, kmono_idx):
    i = kdist_idx[best["Kdist"]]
    j = kmono_idx[best["Kmono"]]
    ax.scatter(
        [j],
        [i],
        marker="*",
        s=260,
        facecolor="#4dd0e1",
        edgecolor="black",
        linewidth=1.0,
        zorder=6,
    )
    info = (
        f"Best: ({best['Kdist']:.1f}, {best['Kmono']:.1f})\n"
        f"quality={best['quality']:.3f}, validity={best['validity_rate']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, pad=3.0),
    )


def save_single_heatmap(ligand, rows, out_prefix: Path, dpi: int, vmin: float, vmax: float):
    mat, kdist_vals, kmono_vals, kdist_idx, kmono_idx = build_matrix(rows, "quality")
    best = choose_best(rows)
    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    im = ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"{ligand} Quality Heatmap (Advantage2_system1.13)")
    stylize_axes(ax, kdist_vals, kmono_vals)
    draw_best_marker(ax, best, kdist_idx, kmono_idx)
    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
    cbar.set_label("Quality = n_success_valid / n_reads")
    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return best, png, pdf


def save_combined_heatmap(rows_by_ligand, best_by_ligand, out_prefix: Path, dpi: int, vmin: float, vmax: float):
    ligands = sorted(rows_by_ligand.keys())
    fig, axes = plt.subplots(1, len(ligands), figsize=(16.0, 6.8), constrained_layout=True)
    if len(ligands) == 1:
        axes = [axes]
    im = None
    for ax, lig in zip(axes, ligands):
        rows = rows_by_ligand[lig]
        mat, kdist_vals, kmono_vals, kdist_idx, kmono_idx = build_matrix(rows, "quality")
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"{lig}")
        stylize_axes(ax, kdist_vals, kmono_vals)
        draw_best_marker(ax, best_by_ligand[lig], kdist_idx, kmono_idx)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Quality = n_success_valid / n_reads")
    fig.suptitle("Penalty Sweep Quality Heatmaps (Advantage2_system1.13)", fontsize=14)
    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def save_optima_csv(best_by_ligand, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["ligand", "Kdist_best", "Kmono_best", "quality_best", "validity_at_best", "mRMSD_at_best"],
        )
        w.writeheader()
        for lig in sorted(best_by_ligand.keys()):
            b = best_by_ligand[lig]
            w.writerow(
                {
                    "ligand": lig,
                    "Kdist_best": f"{b['Kdist']:.1f}",
                    "Kmono_best": f"{b['Kmono']:.1f}",
                    "quality_best": f"{b['quality']:.6f}",
                    "validity_at_best": f"{b['validity_rate']:.6f}",
                    "mRMSD_at_best": f"{b['mRMSD']:.6f}",
                }
            )


def main():
    ap = argparse.ArgumentParser(description="Build publication-quality heatmaps for penalty_sweep_qpu_v113.")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    # Best-of-N aggregation across ALL rep CSVs (no single-CSV / run-tag mode).
    rows, csvs_used = load_all_reps()
    if not rows:
        raise SystemExit("No valid rows found across any case metrics CSV.")

    rows_by_ligand = defaultdict(list)
    for r in rows:
        rows_by_ligand[r["ligand"]].append(r)

    all_q = [r["quality"] for r in rows]
    vmin = 0.0
    vmax = max(all_q) if all_q else 1.0

    pub_dir = FIG_DIR / "pub_v113"
    pub_dir.mkdir(parents=True, exist_ok=True)

    best_by_ligand = {}
    generated = []
    for lig in sorted(rows_by_ligand.keys()):
        prefix = pub_dir / f"quality_heatmap_{lig}"
        best, png, pdf = save_single_heatmap(
            ligand=lig,
            rows=rows_by_ligand[lig],
            out_prefix=prefix,
            dpi=args.dpi,
            vmin=vmin,
            vmax=vmax,
        )
        best_by_ligand[lig] = best
        generated.extend([png, pdf])

    comb_prefix = pub_dir / "quality_heatmap_combined"
    comb_png, comb_pdf = save_combined_heatmap(
        rows_by_ligand=rows_by_ligand,
        best_by_ligand=best_by_ligand,
        out_prefix=comb_prefix,
        dpi=args.dpi,
        vmin=vmin,
        vmax=vmax,
    )
    generated.extend([comb_png, comb_pdf])

    opt_csv = pub_dir / "ligand_optima.csv"
    save_optima_csv(best_by_ligand, opt_csv)
    generated.append(opt_csv)

    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
