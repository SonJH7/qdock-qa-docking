#!/usr/bin/env python3
"""Statistical/robustness analysis for v113 penalty sweeps (r1~r5).

Outputs:
- results/penalty_quality_topk_by_ligand_5run.csv
- results/penalty_quality_top1_vs_top2_stats_5run.csv
- report/penalty_quality_stats_5run.md
- figures/penalty_quality_mean_std_scatter_{ligand}_5run.png
- figures/penalty_quality_top10_errorbar_{ligand}_5run.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class PairKey:
    ligand: str
    kdist: float
    kmono: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Analyze v113 penalty sweep robustness/statistics")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "penalty_sweep_qpu_v113" / "results",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "penalty_sweep_qpu_v113" / "figures",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=repo_root / "penalty_sweep_qpu_v113" / "report",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260331)
    return p.parse_args()


def discover_case_metric_files(results_dir: Path) -> list[Path]:
    files = sorted(results_dir.glob("penalty_sweep_case_metrics_*.csv"))
    if not files:
        raise FileNotFoundError(f"No case metrics found in {results_dir}")
    return files


def read_quality_by_pair(files: list[Path]) -> tuple[list[str], dict[PairKey, dict[str, float]]]:
    run_tags: list[str] = []
    table: dict[PairKey, dict[str, float]] = {}

    for f in files:
        run_tag = f.name.replace("penalty_sweep_case_metrics_", "").replace(".csv", "")
        run_tags.append(run_tag)
        with f.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                if (row.get("status") or "").strip().lower() != "ok":
                    continue
                ligand = (row.get("ligand") or "").strip().lower()
                if ligand not in ("3nq9", "4jsz"):
                    continue
                key = PairKey(
                    ligand=ligand,
                    kdist=float(row["Kdist"]),
                    kmono=float(row["Kmono"]),
                )
                q = float(row["quality"])
                table.setdefault(key, {})[run_tag] = q

    return run_tags, table


def complete_pairs_only(
    run_tags: list[str],
    by_pair: dict[PairKey, dict[str, float]],
) -> dict[PairKey, np.ndarray]:
    out: dict[PairKey, np.ndarray] = {}
    for key, m in by_pair.items():
        if all(rt in m for rt in run_tags):
            out[key] = np.array([float(m[rt]) for rt in run_tags], dtype=float)
    return out


def summarize_pairs(complete: dict[PairKey, np.ndarray]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for key, arr in complete.items():
        rows.append(
            {
                "ligand": key.ligand,
                "kdist": key.kdist,
                "kmono": key.kmono,
                "mean_quality": float(np.mean(arr)),
                "std_quality": float(np.std(arr, ddof=0)),
                "min_quality": float(np.min(arr)),
                "max_quality": float(np.max(arr)),
                "q25_quality": float(np.percentile(arr, 25)),
                "q50_quality": float(np.percentile(arr, 50)),
                "q75_quality": float(np.percentile(arr, 75)),
                "n_runs": int(arr.size),
            }
        )
    return rows


def rank_rows(rows: list[dict[str, float | str]], ligand: str) -> list[dict[str, float | str]]:
    sub = [r for r in rows if r["ligand"] == ligand]
    sub.sort(
        key=lambda r: (
            float(r["mean_quality"]),
            -float(r["std_quality"]),
            -float(r["kdist"]),
            -float(r["kmono"]),
        ),
        reverse=True,
    )
    return sub


def paired_bootstrap_ci_mean_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = x.size
    d = x - y
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(d[idx]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def stats_top1_vs_top2(
    ranked: list[dict[str, float | str]],
    complete: dict[PairKey, np.ndarray],
    bootstrap: int,
    seed: int,
) -> dict[str, float | str]:
    if len(ranked) < 2:
        raise ValueError("Need at least two pairs for top1-vs-top2 comparison")
    r1 = ranked[0]
    r2 = ranked[1]
    k1 = PairKey(str(r1["ligand"]), float(r1["kdist"]), float(r1["kmono"]))
    k2 = PairKey(str(r2["ligand"]), float(r2["kdist"]), float(r2["kmono"]))
    x = complete[k1]
    y = complete[k2]
    diff = x - y

    t_res = stats.ttest_rel(x, y, alternative="two-sided")
    p_t = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else math.nan

    nz = int(np.sum(np.abs(diff) > 1e-12))
    if nz >= 1:
        try:
            w_res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            p_w = float(w_res.pvalue)
        except ValueError:
            p_w = math.nan
    else:
        p_w = math.nan

    ci_lo, ci_hi = paired_bootstrap_ci_mean_diff(x, y, n_boot=bootstrap, seed=seed)
    mean_diff = float(np.mean(diff))

    return {
        "ligand": str(r1["ligand"]),
        "top1_kdist": float(r1["kdist"]),
        "top1_kmono": float(r1["kmono"]),
        "top1_mean_quality": float(r1["mean_quality"]),
        "top1_std_quality": float(r1["std_quality"]),
        "top2_kdist": float(r2["kdist"]),
        "top2_kmono": float(r2["kmono"]),
        "top2_mean_quality": float(r2["mean_quality"]),
        "top2_std_quality": float(r2["std_quality"]),
        "mean_diff_top1_minus_top2": mean_diff,
        "paired_ttest_pvalue": p_t,
        "wilcoxon_pvalue": p_w,
        "bootstrap_ci95_lo": ci_lo,
        "bootstrap_ci95_hi": ci_hi,
        "n_runs": int(x.size),
        "n_nonzero_diffs": nz,
    }


def save_csv(path: Path, rows: list[dict[str, float | str]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    if field_order is None:
        field_order = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=field_order)
        wr.writeheader()
        wr.writerows(rows)


def plot_mean_std_scatter(
    ligand: str,
    ranked: list[dict[str, float | str]],
    out_path: Path,
) -> None:
    xs = np.array([float(r["mean_quality"]) for r in ranked], dtype=float)
    ys = np.array([float(r["std_quality"]) for r in ranked], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.scatter(xs, ys, s=14, alpha=0.45, color="#5B6C8F")

    for i, color in zip([0, 1], ["#C00000", "#0072B2"]):
        if i < len(ranked):
            r = ranked[i]
            x = float(r["mean_quality"])
            y = float(r["std_quality"])
            ax.scatter([x], [y], s=70, color=color, zorder=4)
            ax.text(
                x + 0.002,
                y + 0.002,
                f"Top{i+1} ({float(r['kdist']):.1f},{float(r['kmono']):.1f})",
                fontsize=8,
                color=color,
            )

    ax.set_title(f"{ligand} quality robustness (mean vs std, n=5)")
    ax.set_xlabel("Mean quality over 5 runs")
    ax.set_ylabel("Std(quality) over 5 runs")
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_topk_errorbar(
    ligand: str,
    ranked: list[dict[str, float | str]],
    k: int,
    out_path: Path,
) -> None:
    top = ranked[:k]
    labels = [f"({float(r['kdist']):.1f},{float(r['kmono']):.1f})" for r in top]
    means = np.array([float(r["mean_quality"]) for r in top], dtype=float)
    stds = np.array([float(r["std_quality"]) for r in top], dtype=float)
    x = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(max(9.0, 0.85 * len(top)), 4.8))
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=3, lw=1.2, color="#2C7FB8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Quality (mean ± std)")
    ax.set_title(f"{ligand} top-{len(top)} quality candidates (5-run)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_markdown_report(
    run_tags: list[str],
    stat_rows: list[dict[str, float | str]],
    topk_path: Path,
    stat_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Penalty Quality Statistical Validation (v113, 5 runs)")
    lines.append("")
    lines.append("## Runs used")
    for t in run_tags:
        lines.append(f"- `{t}`")
    lines.append("")
    lines.append("## Top1 vs Top2 significance")
    for s in stat_rows:
        lig = str(s["ligand"])
        lines.append(f"### {lig}")
        lines.append(
            f"- Top1: (`{s['top1_kdist']:.1f}`, `{s['top1_kmono']:.1f}`), "
            f"mean={s['top1_mean_quality']:.6f}, std={s['top1_std_quality']:.6f}"
        )
        lines.append(
            f"- Top2: (`{s['top2_kdist']:.1f}`, `{s['top2_kmono']:.1f}`), "
            f"mean={s['top2_mean_quality']:.6f}, std={s['top2_std_quality']:.6f}"
        )
        lines.append(
            f"- mean diff (Top1-Top2)={s['mean_diff_top1_minus_top2']:.6f}, "
            f"paired t-test p={s['paired_ttest_pvalue']:.6g}, "
            f"wilcoxon p={s['wilcoxon_pvalue']:.6g}"
        )
        lines.append(
            f"- bootstrap 95% CI of mean diff: "
            f"[{s['bootstrap_ci95_lo']:.6f}, {s['bootstrap_ci95_hi']:.6f}]"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Top-k table: `{topk_path}`")
    lines.append(f"- Stats table: `{stat_path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    files = discover_case_metric_files(args.results_dir)
    run_tags, by_pair = read_quality_by_pair(files)
    complete = complete_pairs_only(run_tags, by_pair)
    summary = summarize_pairs(complete)

    topk_rows: list[dict[str, float | str]] = []
    stat_rows: list[dict[str, float | str]] = []

    for lig in ("3nq9", "4jsz"):
        ranked = rank_rows(summary, ligand=lig)
        if not ranked:
            continue
        for rank, row in enumerate(ranked[: args.top_k], start=1):
            rr = dict(row)
            rr["rank"] = rank
            topk_rows.append(rr)

        stat_rows.append(
            stats_top1_vs_top2(
                ranked=ranked,
                complete=complete,
                bootstrap=args.bootstrap,
                seed=args.seed + (1 if lig == "4jsz" else 0),
            )
        )

        plot_mean_std_scatter(
            ligand=lig,
            ranked=ranked,
            out_path=args.figures_dir / f"penalty_quality_mean_std_scatter_{lig}_5run.png",
        )
        plot_topk_errorbar(
            ligand=lig,
            ranked=ranked,
            k=args.top_k,
            out_path=args.figures_dir / f"penalty_quality_top{args.top_k}_errorbar_{lig}_5run.png",
        )

    topk_csv = args.results_dir / "penalty_quality_topk_by_ligand_5run.csv"
    stat_csv = args.results_dir / "penalty_quality_top1_vs_top2_stats_5run.csv"
    save_csv(topk_csv, topk_rows)
    save_csv(stat_csv, stat_rows)

    report_md = build_markdown_report(
        run_tags=run_tags,
        stat_rows=stat_rows,
        topk_path=topk_csv,
        stat_path=stat_csv,
    )
    report_path = args.report_dir / "penalty_quality_stats_5run.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    print(f"[done] topk:  {topk_csv}")
    print(f"[done] stats: {stat_csv}")
    print(f"[done] report:{report_path}")


if __name__ == "__main__":
    main()
