#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LIGAND_TAG_TO_NAME = {
    "q3": "3nq9",
    "q4": "4jsz",
}

METHOD_COLORS = {
    "bo": "#1f77b4",
    "random": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BO vs Random comparison bundle (v113)")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--out-root", type=Path, default=Path("BO_vs_Random_v113"))
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--threshold-delta", type=float, default=0.02)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--use-matched-runs", action="store_true", default=True)
    p.add_argument(
        "--exclude-run-name",
        action="append",
        default=[],
        help="Run directory basename to exclude. Can be repeated.",
    )
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()


def discover_run_dirs(repo_root: Path) -> Dict[Tuple[str, str], List[Path]]:
    patterns = {
        ("q3", "bo"): "BO_Online_v113_b100_q3_n20_x0.1*",
        ("q3", "random"): "BO_Online_v113_b100_q3_random_n20_x0.1*",
        ("q4", "bo"): "BO_Online_v113_b100_q4_n20_x0.1*",
        ("q4", "random"): "BO_Online_v113_b100_q4_random_n20_x0.1*",
    }
    out: Dict[Tuple[str, str], List[Path]] = {}
    for key, pat in patterns.items():
        dirs = sorted([p for p in repo_root.glob(pat) if p.is_dir()])
        if key[1] == "bo":
            dirs = [d for d in dirs if "random" not in d.name]
        out[key] = dirs
    return out


def get_trace_path(run_dir: Path, allow_partial: bool) -> Tuple[Path | None, bool]:
    full = run_dir / "results" / "online_bo_trace.csv"
    partial = run_dir / "results" / "online_bo_trace_partial.csv"
    if full.exists():
        return full, False
    if allow_partial and partial.exists():
        return partial, True
    return None, False


def load_run_inventory(
    repo_root: Path,
    discovered: Dict[Tuple[str, str], List[Path]],
    budget: int,
    allow_partial: bool,
    exclude_run_names: List[str],
) -> pd.DataFrame:
    rows = []
    for (lig_tag, method), run_dirs in discovered.items():
        for d in run_dirs:
            if d.name in exclude_run_names:
                rows.append(
                    {
                        "ligand_tag": lig_tag,
                        "ligand": LIGAND_TAG_TO_NAME[lig_tag],
                        "method": method,
                        "run_dir": str(d),
                        "run_name": d.name,
                        "trace_path": "",
                        "is_partial_file": False,
                        "is_complete": False,
                        "step_max": np.nan,
                        "rows": 0,
                        "note": "excluded_by_name",
                    }
                )
                continue

            trace_path, is_partial_file = get_trace_path(d, allow_partial)
            if trace_path is None:
                rows.append(
                    {
                        "ligand_tag": lig_tag,
                        "ligand": LIGAND_TAG_TO_NAME[lig_tag],
                        "method": method,
                        "run_dir": str(d),
                        "run_name": d.name,
                        "trace_path": "",
                        "is_partial_file": False,
                        "is_complete": False,
                        "step_max": np.nan,
                        "rows": 0,
                        "note": "missing_trace",
                    }
                )
                continue

            try:
                df = pd.read_csv(trace_path)
                step_max = int(df["step"].max()) if len(df) > 0 and "step" in df.columns else 0
                is_complete = (not is_partial_file) and (step_max >= budget)
                rows.append(
                    {
                        "ligand_tag": lig_tag,
                        "ligand": LIGAND_TAG_TO_NAME[lig_tag],
                        "method": method,
                        "run_dir": str(d),
                        "run_name": d.name,
                        "trace_path": str(trace_path),
                        "is_partial_file": bool(is_partial_file),
                        "is_complete": bool(is_complete),
                        "step_max": step_max,
                        "rows": int(len(df)),
                        "note": "ok",
                    }
                )
            except Exception as e:  # noqa: BLE001
                rows.append(
                    {
                        "ligand_tag": lig_tag,
                        "ligand": LIGAND_TAG_TO_NAME[lig_tag],
                        "method": method,
                        "run_dir": str(d),
                        "run_name": d.name,
                        "trace_path": str(trace_path),
                        "is_partial_file": bool(is_partial_file),
                        "is_complete": False,
                        "step_max": np.nan,
                        "rows": 0,
                        "note": f"read_error:{e}",
                    }
                )
    return pd.DataFrame(rows)


def select_runs(inv: pd.DataFrame, use_matched_runs: bool) -> pd.DataFrame:
    selected = []
    inv = inv.copy()
    inv["selected"] = False

    for lig_tag in ["q3", "q4"]:
        bo = inv[(inv.ligand_tag == lig_tag) & (inv.method == "bo") & (inv.is_complete)]
        rd = inv[(inv.ligand_tag == lig_tag) & (inv.method == "random") & (inv.is_complete)]

        bo_names = sorted(bo["run_name"].tolist())
        rd_names = sorted(rd["run_name"].tolist())

        if use_matched_runs:
            n = min(len(bo_names), len(rd_names))
            bo_sel = set(bo_names[:n])
            rd_sel = set(rd_names[:n])
        else:
            bo_sel = set(bo_names)
            rd_sel = set(rd_names)

        mask_bo = (inv.ligand_tag == lig_tag) & (inv.method == "bo") & inv.run_name.isin(bo_sel)
        mask_rd = (inv.ligand_tag == lig_tag) & (inv.method == "random") & inv.run_name.isin(rd_sel)
        inv.loc[mask_bo | mask_rd, "selected"] = True

        selected.append(
            {
                "ligand_tag": lig_tag,
                "ligand": LIGAND_TAG_TO_NAME[lig_tag],
                "bo_complete": len(bo_names),
                "random_complete": len(rd_names),
                "bo_selected": len(bo_sel),
                "random_selected": len(rd_sel),
                "matched": bool(use_matched_runs),
            }
        )

    return inv, pd.DataFrame(selected)


def load_selected_traces(inv_selected: pd.DataFrame) -> pd.DataFrame:
    parts = []
    rows = inv_selected[inv_selected["selected"]].copy()
    for _, r in rows.iterrows():
        df = pd.read_csv(r["trace_path"]).copy()
        keep = [
            "step",
            "kdist",
            "kmono",
            "objective_value",
            "best_so_far",
            "q_3nq9",
            "q_4jsz",
            "v_3nq9",
            "v_4jsz",
            "success_3nq9",
            "success_4jsz",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()
        df["run_name"] = r["run_name"]
        df["ligand_tag"] = r["ligand_tag"]
        df["ligand"] = r["ligand"]
        df["method"] = r["method"]
        parts.append(df)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["ligand_tag", "method", "run_name", "step"]).reset_index(drop=True)
    return out


def load_oracle_mean_map(repo_root: Path) -> pd.DataFrame:
    files = sorted((repo_root / "penalty_sweep_qpu_v113" / "results").glob("penalty_sweep_case_metrics_psweep*.csv"))
    if not files:
        raise FileNotFoundError("No penalty sweep case metrics found")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df = df[df["status"] == "ok"].copy()
        frames.append(df[["ligand", "Kdist", "Kmono", "quality", "run_tag"]])

    all_df = pd.concat(frames, ignore_index=True)
    mean_map = (
        all_df.groupby(["ligand", "Kdist", "Kmono"], as_index=False)
        .agg(quality_mean=("quality", "mean"), quality_std=("quality", "std"), n_runs=("quality", "size"))
    )
    mean_map["quality_std"] = mean_map["quality_std"].fillna(0.0)
    return mean_map


def compute_thresholds(mean_map: pd.DataFrame, delta: float) -> pd.DataFrame:
    rows = []
    for ligand, sub in mean_map.groupby("ligand"):
        best = sub.sort_values("quality_mean", ascending=False).iloc[0]
        rows.append(
            {
                "ligand": ligand,
                "oracle_kdist": float(best["Kdist"]),
                "oracle_kmono": float(best["Kmono"]),
                "oracle_quality": float(best["quality_mean"]),
                "threshold": float(best["quality_mean"] - delta),
                "delta": float(delta),
            }
        )
    return pd.DataFrame(rows)


def compute_recovery_stats(traces: pd.DataFrame) -> pd.DataFrame:
    g = traces.groupby(["ligand_tag", "ligand", "method", "step"], as_index=False).agg(
        mean_best_so_far=("best_so_far", "mean"),
        std_best_so_far=("best_so_far", "std"),
        n_runs=("best_so_far", "size"),
    )
    g["std_best_so_far"] = g["std_best_so_far"].fillna(0.0)
    g["ci95"] = 1.96 * g["std_best_so_far"] / np.sqrt(g["n_runs"].clip(lower=1))
    return g


def compute_run_level_metrics(
    traces: pd.DataFrame,
    thresholds: pd.DataFrame,
    budget: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    th_map = thresholds.set_index("ligand")["threshold"].to_dict()

    final_rows = []
    hit_rows = []

    for (ligand_tag, ligand, method, run_name), sub in traces.groupby(
        ["ligand_tag", "ligand", "method", "run_name"], as_index=False
    ):
        s = sub.sort_values("step")
        final_best = float(s["best_so_far"].iloc[-1])
        final_step = int(s["step"].iloc[-1])
        final_rows.append(
            {
                "ligand_tag": ligand_tag,
                "ligand": ligand,
                "method": method,
                "run_name": run_name,
                "final_step": final_step,
                "final_best_so_far": final_best,
            }
        )

        th = float(th_map[ligand])
        hit = s[s["best_so_far"] >= th]
        if len(hit) > 0:
            hit_step = int(hit["step"].iloc[0])
            hit_flag = True
        else:
            hit_step = int(budget + 1)
            hit_flag = False

        hit_rows.append(
            {
                "ligand_tag": ligand_tag,
                "ligand": ligand,
                "method": method,
                "run_name": run_name,
                "threshold": th,
                "hit_step": hit_step,
                "hit": hit_flag,
            }
        )

    return pd.DataFrame(final_rows), pd.DataFrame(hit_rows)


def _draw_method_box(ax, data_bo, data_rand, labels=("BO", "Random"), title="", ylabel=""):
    bp = ax.boxplot([data_bo, data_rand], labels=list(labels), patch_artist=True, widths=0.58)
    for patch, color in zip(bp["boxes"], [METHOD_COLORS["bo"], METHOD_COLORS["random"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    rng = np.random.default_rng(2026)
    for i, vals in enumerate([data_bo, data_rand], start=1):
        if len(vals) == 0:
            continue
        x = rng.normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, s=22, alpha=0.8, color=[METHOD_COLORS["bo"], METHOD_COLORS["random"]][i - 1], edgecolor="white", linewidth=0.4)
        med = float(np.median(vals))
        ax.text(i, med, f"med={med:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_single_metric_hitstep(hit_df: pd.DataFrame, thresholds: pd.DataFrame, out_path: Path, dpi: int):
    th_map = thresholds.set_index("ligand")["threshold"].to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)

    for ax, lig_tag in zip(axes, ["q3", "q4"]):
        lig = LIGAND_TAG_TO_NAME[lig_tag]
        sub = hit_df[hit_df["ligand_tag"] == lig_tag]
        bo = sub[sub["method"] == "bo"]["hit_step"].to_numpy(dtype=float)
        rd = sub[sub["method"] == "random"]["hit_step"].to_numpy(dtype=float)

        _draw_method_box(
            ax,
            bo,
            rd,
            title=f"{lig} hit-step (threshold={th_map.get(lig, np.nan):.3f})",
            ylabel="Hit step (101 = not hit)",
        )
        ax.set_ylim(0, 105)

    fig.suptitle("Primary Metric: BO vs Random Hit-step", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_panel_2x2(
    recovery: pd.DataFrame,
    final_df: pd.DataFrame,
    hit_df: pd.DataFrame,
    thresholds: pd.DataFrame,
    out_path: Path,
    dpi: int,
):
    th_map = thresholds.set_index("ligand")["threshold"].to_dict()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))

    # A/B: recovery curves
    for ax, lig_tag, ttl in [
        (axes[0, 0], "q3", f"Recovery Curve: {LIGAND_TAG_TO_NAME['q3']}"),
        (axes[0, 1], "q4", f"Recovery Curve: {LIGAND_TAG_TO_NAME['q4']}"),
    ]:
        sub = recovery[recovery["ligand_tag"] == lig_tag]
        for method in ["bo", "random"]:
            m = sub[sub["method"] == method].sort_values("step")
            if m.empty:
                continue
            ax.plot(m["step"], m["mean_best_so_far"], color=METHOD_COLORS[method], linewidth=2.0, label=f"{method.upper()} (n={int(m['n_runs'].iloc[0])})")
            lo = (m["mean_best_so_far"] - m["ci95"]).to_numpy()
            hi = (m["mean_best_so_far"] + m["ci95"]).to_numpy()
            ax.fill_between(m["step"], lo, hi, color=METHOD_COLORS[method], alpha=0.2)

        ax.set_title(ttl)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best-so-far Quality")
        ax.grid(alpha=0.25)
        ax.legend()

    # C: final-best distribution across ligand+method
    ax = axes[1, 0]
    cats = [
        ("q3", "bo", "q3-BO"),
        ("q3", "random", "q3-Random"),
        ("q4", "bo", "q4-BO"),
        ("q4", "random", "q4-Random"),
    ]
    data = [
        final_df[(final_df.ligand_tag == lt) & (final_df.method == md)]["final_best_so_far"].to_numpy(dtype=float)
        for lt, md, _ in cats
    ]
    labels = [lbl for _, _, lbl in cats]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.58)
    colors = [METHOD_COLORS["bo"], METHOD_COLORS["random"], METHOD_COLORS["bo"], METHOD_COLORS["random"]]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.set_title("Final best-so-far (step=100)")
    ax.set_ylabel("Quality")
    ax.grid(alpha=0.25)
    for i, arr in enumerate(data, start=1):
        if len(arr) > 0:
            ax.text(i, float(np.median(arr)), f"{np.median(arr):.3f}", ha="center", va="bottom", fontsize=8)

    # D: hit-step distribution
    ax = axes[1, 1]
    data_hit = [
        hit_df[(hit_df.ligand_tag == lt) & (hit_df.method == md)]["hit_step"].to_numpy(dtype=float)
        for lt, md, _ in cats
    ]
    bp = ax.boxplot(data_hit, labels=labels, patch_artist=True, widths=0.58)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.set_title(
        f"Hit-step (q3 th={th_map.get('3nq9', np.nan):.3f}, q4 th={th_map.get('4jsz', np.nan):.3f})"
    )
    ax.set_ylabel("Hit step (101 = not hit)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)

    fig.suptitle("BO vs Random (v113) Summary", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_overlay(
    traces: pd.DataFrame,
    mean_map: pd.DataFrame,
    thresholds: pd.DataFrame,
    out_dir: Path,
    dpi: int,
):
    best_map = thresholds.set_index("ligand")

    for lig_tag in ["q3", "q4"]:
        lig = LIGAND_TAG_TO_NAME[lig_tag]
        sub_map = mean_map[mean_map["ligand"] == lig].copy()

        x_vals = sorted(sub_map["Kdist"].unique())
        y_vals = sorted(sub_map["Kmono"].unique())
        pivot = sub_map.pivot(index="Kmono", columns="Kdist", values="quality_mean")
        pivot = pivot.reindex(index=y_vals, columns=x_vals)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharex=True, sharey=True)
        for ax, method in zip(axes, ["bo", "random"]):
            im = ax.imshow(
                pivot.values,
                origin="lower",
                aspect="auto",
                extent=[min(x_vals) - 0.2, max(x_vals) + 0.2, min(y_vals) - 0.5, max(y_vals) + 0.5],
                cmap="viridis",
                alpha=0.95,
            )

            q = traces[(traces["ligand_tag"] == lig_tag) & (traces["method"] == method)]
            cnt = q.groupby(["kdist", "kmono"], as_index=False).size().rename(columns={"size": "visit_count"})
            if len(cnt) > 0:
                ax.scatter(
                    cnt["kdist"],
                    cnt["kmono"],
                    s=22 + 8 * cnt["visit_count"].to_numpy(),
                    color="#ff4d4d",
                    alpha=0.35,
                    edgecolor="white",
                    linewidth=0.4,
                    label="queried pairs (size~count)",
                )

            b = best_map.loc[lig]
            ax.scatter([b["oracle_kdist"]], [b["oracle_kmono"]], marker="*", s=220, color="gold", edgecolor="black", linewidth=0.6, label="oracle best")

            n_runs = q["run_name"].nunique()
            total_q = len(q)
            ax.set_title(f"{lig} | {method.upper()} overlay (runs={n_runs}, queries={total_q})")
            ax.set_xlabel("Kdist")
            ax.grid(alpha=0.18)

        axes[0].set_ylabel("Kmono")
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels, fontsize=8, loc="upper left")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
        cbar.set_label("Oracle mean quality (5-run)")

        fig.tight_layout()
        out = out_dir / f"heatmap_overlay_{lig}_bo_vs_random.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def write_report(
    out_report: Path,
    selection_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    final_df: pd.DataFrame,
    hit_df: pd.DataFrame,
):
    out_report.parent.mkdir(parents=True, exist_ok=True)

    def _df_to_markdown_safe(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    vals.append("")
                elif isinstance(v, (float, np.floating)):
                    vals.append(f"{float(v):.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _iqr(x: pd.Series) -> float:
        return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else np.nan

    final_stats = final_df.groupby(["ligand", "method"], as_index=False).agg(
        n=("final_best_so_far", "size"),
        median_final_best=("final_best_so_far", "median"),
        iqr_final_best=("final_best_so_far", _iqr),
        mean_final_best=("final_best_so_far", "mean"),
    )

    hit_stats = hit_df.groupby(["ligand", "method"], as_index=False).agg(
        n=("hit_step", "size"),
        median_hit_step=("hit_step", "median"),
        iqr_hit_step=("hit_step", _iqr),
        hit_rate=("hit", "mean"),
    )

    with out_report.open("w", encoding="utf-8") as f:
        f.write("# BO vs Random Comparison Bundle (v113)\n\n")
        f.write("## Selection Summary\n\n")
        f.write(_df_to_markdown_safe(selection_summary))
        f.write("\n\n## Oracle Thresholds\n\n")
        f.write(_df_to_markdown_safe(thresholds))
        f.write("\n\n## Final Best Summary\n\n")
        f.write(_df_to_markdown_safe(final_stats))
        f.write("\n\n## Hit-step Summary\n\n")
        f.write(_df_to_markdown_safe(hit_stats))
        f.write("\n")


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_root = (repo_root / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root.resolve()

    for sub in [
        out_root / "scripts",
        out_root / "data",
        out_root / "report",
        out_root / "figures" / "single_metric",
        out_root / "figures" / "panel_2x2",
        out_root / "figures" / "heatmap_overlay",
    ]:
        sub.mkdir(parents=True, exist_ok=True)

    discovered = discover_run_dirs(repo_root)
    inv = load_run_inventory(
        repo_root,
        discovered,
        args.budget,
        args.allow_partial,
        args.exclude_run_name,
    )
    inv.to_csv(out_root / "data" / "run_inventory.csv", index=False)

    inv_sel, sel_summary = select_runs(inv, args.use_matched_runs)
    inv_sel.to_csv(out_root / "data" / "selected_runs.csv", index=False)
    sel_summary.to_csv(out_root / "data" / "selection_summary.csv", index=False)

    traces = load_selected_traces(inv_sel)
    if traces.empty:
        raise RuntimeError("No selected traces. Check run folders or selection criteria.")
    traces.to_csv(out_root / "data" / "selected_traces_long.csv", index=False)

    mean_map = load_oracle_mean_map(repo_root)
    mean_map.to_csv(out_root / "data" / "oracle_quality_map_5run.csv", index=False)

    thresholds = compute_thresholds(mean_map, args.threshold_delta)
    thresholds.to_csv(out_root / "data" / "oracle_thresholds.csv", index=False)

    recovery = compute_recovery_stats(traces)
    recovery.to_csv(out_root / "data" / "recovery_stats.csv", index=False)

    final_df, hit_df = compute_run_level_metrics(traces, thresholds, args.budget)
    final_df.to_csv(out_root / "data" / "final_best_per_run.csv", index=False)
    hit_df.to_csv(out_root / "data" / "hit_step_per_run.csv", index=False)

    plot_single_metric_hitstep(
        hit_df,
        thresholds,
        out_root / "figures" / "single_metric" / "hit_step_primary_bo_vs_random.png",
        dpi=args.dpi,
    )
    plot_panel_2x2(
        recovery,
        final_df,
        hit_df,
        thresholds,
        out_root / "figures" / "panel_2x2" / "bo_vs_random_summary_2x2.png",
        dpi=args.dpi,
    )
    plot_heatmap_overlay(
        traces,
        mean_map,
        thresholds,
        out_root / "figures" / "heatmap_overlay",
        dpi=args.dpi,
    )

    write_report(out_root / "report" / "bo_vs_random_v113_summary.md", sel_summary, thresholds, final_df, hit_df)

    run_cfg = {
        "repo_root": str(repo_root),
        "out_root": str(out_root),
        "budget": args.budget,
        "threshold_delta": args.threshold_delta,
        "allow_partial": args.allow_partial,
        "use_matched_runs": args.use_matched_runs,
        "exclude_run_name": args.exclude_run_name,
    }
    (out_root / "report" / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print("Wrote bundle to:", out_root)
    print("- data/run_inventory.csv")
    print("- data/selected_runs.csv")
    print("- data/recovery_stats.csv")
    print("- data/final_best_per_run.csv")
    print("- data/hit_step_per_run.csv")
    print("- figures/single_metric/hit_step_primary_bo_vs_random.png")
    print("- figures/panel_2x2/bo_vs_random_summary_2x2.png")
    print("- figures/heatmap_overlay/heatmap_overlay_3nq9_bo_vs_random.png")
    print("- figures/heatmap_overlay/heatmap_overlay_4jsz_bo_vs_random.png")
    print("- report/bo_vs_random_v113_summary.md")


if __name__ == "__main__":
    main()
