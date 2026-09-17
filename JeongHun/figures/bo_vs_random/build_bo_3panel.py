"""Build the 3-panel BO-vs-Random figure (Figure 6 in new.tex).

Panel layout:
  [ Resource Efficiency bar ] [ 3NQ9 best-so-far curves ] [ 4JSZ best-so-far curves ]

Resource bar: compares the measured total QPU access time of the full
five-repetition penalty sweep (4,000 target-specific calls) with the measured
mean per-run totals for BO and random search. Curves summarize the 20-run
comparison (five runs per policy and target) as best-so-far Quality vs.
evaluation step, with the highest five-run cell-level mean from the reference
sweep drawn as a dashed horizontal line.

The trajectory and reference-line values are sourced from the experiment CSVs
under `tables/bo_vs_random/` and the upstream penalty-sweep case-metrics files.
The resource panel is computed directly from the preregistered run traces and
the five penalty-sweep case-metrics files.

Outputs:
  figures/bo_vs_random/bo_vs_random_3panel.{png,pdf,svg}

Style (2026-08-05, QST publication pass): axis titles / suptitle removed;
panels carry (a)/(b)/(c) sub-labels; series identity via legend (target name
moved into the legend title); all font sizes enlarged for print legibility.
The curve-panel data are unchanged; the resource panel uses the measured
campaign totals described above.
"""
from __future__ import annotations

import csv
import os
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---- Publication style (enlarged fonts; no in-axes titles) ----
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 11.5,
    "legend.title_fontsize": 12.5,
    "font.size": 11.5,
})


def panel_label(ax, s):
    """(a)/(b)/(c) sub-panel label at the top-left outside corner."""
    return  # (a)/(b)/(c) markers moved to the LaTeX caption per reviewer request


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

OUT_PNG = os.path.join(THIS_DIR, "bo_vs_random_3panel.png")
OUT_PDF = os.path.join(THIS_DIR, "bo_vs_random_3panel.pdf")
OUT_SVG = os.path.join(THIS_DIR, "bo_vs_random_3panel.svg")

# ---------- Constants (derived from the experiment manifests) -----------
RESOURCE_GROUPS = (
    ("3nq9", "bo"),
    ("3nq9", "random"),
    ("4jsz", "bo"),
    ("4jsz", "random"),
)

# Reference line: the largest per-cell 5-run MEAN Quality over the brute-force
# penalty grid (max over cells of the mean over the 5 QPU repetitions), i.e. the
# best cell-level mean rather than the optimistic best-of-5 ceiling.
# Source: penalty_sweep_qpu_v113 case_metrics CSVs (verified 2026-08-06):
#   3NQ9 cell (6.5, 2.0) = 0.2546 ; 4JSZ cell (7.5, 13.0) = 0.2484.
ORACLE_3NQ9 = 0.2546
ORACLE_4JSZ = 0.2484


def read_recovery(path):
    """Return mapping (ligand, method) -> (steps, mean, ci95) for the curve."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    out = {}
    for lig in ("3nq9", "4jsz"):
        for meth in ("bo", "random"):
            sel = [x for x in rows
                   if x["ligand"] == lig and x["method"] == meth]
            sel.sort(key=lambda x: int(x["step"]))
            s = np.array([int(x["step"]) for x in sel])
            m = np.array([float(x["mean_best_so_far"]) for x in sel])
            ci = np.array([float(x["ci95"]) for x in sel])
            out[(lig, meth)] = (s, m, ci)
    return out


def measured_run_seconds(trace_csv):
    """Return measured QPU access seconds for one 100-evaluation run."""
    with open(trace_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise RuntimeError(
            f"Expected 100 evaluations in {trace_csv}, found {len(rows)}"
        )
    required = ("qpu_access_time_3nq9", "qpu_access_time_4jsz")
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise RuntimeError(f"Missing access-time columns in {trace_csv}: {missing}")
    access_microseconds = sum(
        float(row["qpu_access_time_3nq9"])
        + float(row["qpu_access_time_4jsz"])
        for row in rows
    )
    return access_microseconds / 1_000_000.0


def load_policy_run_seconds():
    """Load the ten BO and ten random per-run access-time totals."""
    campaign_root = os.path.join(REPO_ROOT, "BO_Online_prereg_20260804")
    values = {"bo": [], "random": []}
    for target, policy in RESOURCE_GROUPS:
        prefix = f"{target}_{policy}_s"
        run_dirs = sorted(
            os.path.join(campaign_root, name)
            for name in os.listdir(campaign_root)
            if name.startswith(prefix)
            and os.path.isdir(os.path.join(campaign_root, name))
        )
        if len(run_dirs) != 5:
            raise RuntimeError(
                f"Expected five {target}/{policy} runs under {campaign_root}, "
                f"found {len(run_dirs)}"
            )
        for run_dir in run_dirs:
            trace_csv = os.path.join(run_dir, "results", "online_bo_trace.csv")
            if not os.path.isfile(trace_csv):
                raise FileNotFoundError(trace_csv)
            values[policy].append(measured_run_seconds(trace_csv))
    for policy, policy_values in values.items():
        if len(policy_values) != 10:
            raise RuntimeError(
                f"Expected ten runs for {policy}, found {len(policy_values)}"
            )
    return values


def load_full_penalty_sweep_seconds():
    """Return measured QPU access seconds across the full five-rep sweep."""
    results_dir = os.path.join(REPO_ROOT, "penalty_sweep_qpu_v113", "results")
    metric_files = sorted(
        os.path.join(results_dir, name)
        for name in os.listdir(results_dir)
        if name.startswith("penalty_sweep_case_metrics_psweep")
        and name.endswith(".csv")
    )
    if len(metric_files) != 5:
        raise RuntimeError(
            f"Expected five penalty-sweep repetitions under {results_dir}, "
            f"found {len(metric_files)}"
        )
    total_seconds = 0.0
    total_calls = 0
    for metric_file in metric_files:
        with open(metric_file, newline="") as handle:
            rows = list(csv.DictReader(handle))
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        target_counts = {
            target: sum(row.get("ligand") == target for row in ok_rows)
            for target in ("3nq9", "4jsz")
        }
        if len(ok_rows) != 800 or target_counts != {"3nq9": 400, "4jsz": 400}:
            raise RuntimeError(
                f"Expected 400 successful calls per target in {metric_file}; "
                f"found {target_counts}"
            )
        total_seconds += sum(float(row["qpu_access_sec"]) for row in ok_rows)
        total_calls += len(ok_rows)
    if total_calls != 4_000:
        raise RuntimeError(f"Expected 4,000 full-sweep calls, found {total_calls}")
    return total_seconds


def panel_resource(ax, full_s, policy_seconds):
    """Measured QPU-access resource comparison."""
    bo_s = mean(policy_seconds["bo"])
    random_s = mean(policy_seconds["random"])
    labels = [
        "Full penalty sweep\n5 repetitions",
        "BO\nmean per run\n($n$=10)",
        "Random\nmean per run\n($n$=10)",
    ]
    values = [full_s, bo_s, random_s]
    colors = ["#7d7d7d", "#1f4e8a", "#c92a2a"]
    annotations = [
        f"{full_s:.1f} s\n(100%)",
        f"{bo_s:.1f} s\n({100.0 * bo_s / full_s:.1f}%)",
        f"{random_s:.1f} s\n({100.0 * random_s / full_s:.1f}%)",
    ]

    bars = ax.bar(range(len(labels)), values, color=colors,
                  edgecolor="#222", linewidth=0.8, width=0.68, zorder=3)
    for bar, value, annotation in zip(bars, values, annotations):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 18.0,
                annotation, ha="center", va="bottom", fontsize=13.0,
                fontweight="bold", color="#172033")
    ax.set_ylim(0, full_s * 1.20)
    ax.set_ylabel("Measured QPU access time (s)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_curve(ax, recov, lig, oracle, target_label, legend_in_title=True):
    s_bo, m_bo, ci_bo = recov[(lig, "bo")]
    s_rs, m_rs, ci_rs = recov[(lig, "random")]
    ax.plot(s_bo, m_bo, color="#1f4e8a", lw=2.0, label="BO")
    ax.fill_between(s_bo, m_bo - ci_bo, m_bo + ci_bo,
                    color="#1f4e8a", alpha=0.18)
    ax.plot(s_rs, m_rs, color="#c92a2a", lw=2.0, label="Random")
    ax.fill_between(s_rs, m_rs - ci_rs, m_rs + ci_rs,
                    color="#c92a2a", alpha=0.15)
    # Best cell-level mean line (max over cells of the 5-run mean Quality).
    ax.axhline(oracle, color="#222", lw=1.4, linestyle="--",
               label="Best cell-level mean")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Evaluated points (budget = 100)")
    ax.set_ylabel("Best-so-far Quality")
    # Target identity carried by the legend title (in place of an axes title).
    ax.legend(loc="lower right", title=(target_label if legend_in_title else None),
              frameon=True, framealpha=0.92, edgecolor="#aaa")
    ax.grid(alpha=0.25, linestyle=":")
    for s in ax.spines.values():
        s.set_color("#444"); s.set_linewidth(0.7)


def main():
    recov = read_recovery(os.path.join(ROOT,
                                       "tables/bo_vs_random/recovery_stats.csv"))
    policy_seconds = load_policy_run_seconds()
    full_sweep_seconds = load_full_penalty_sweep_seconds()

    fig = plt.figure(figsize=(16.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.4, 1.4],
                          wspace=0.30, left=0.08, right=0.985,
                          top=0.93, bottom=0.24)

    ax0 = fig.add_subplot(gs[0, 0])
    panel_resource(ax0, full_sweep_seconds, policy_seconds)
    panel_label(ax0, "(a)")

    ax1 = fig.add_subplot(gs[0, 1])
    panel_curve(ax1, recov, "3nq9", ORACLE_3NQ9, "3NQ9")
    panel_label(ax1, "(b)")

    ax2 = fig.add_subplot(gs[0, 2])
    panel_curve(ax2, recov, "4jsz", ORACLE_4JSZ, "4JSZ")
    panel_label(ax2, "(c)")

    for _ax in fig.axes:
        for _l in _ax.get_xticklabels() + _ax.get_yticklabels():
            _l.set_fontweight("bold")
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)

    # --- individual per-panel PDFs for the subfigure layout ---
    fr, ar = plt.subplots(figsize=(6.8, 5.5))
    panel_resource(ar, full_sweep_seconds, policy_seconds)
    fr.subplots_adjust(left=0.18, right=0.98, bottom=0.23, top=0.94)
    for _l in ar.get_xticklabels() + ar.get_yticklabels():
        _l.set_fontweight("bold")
    fr.savefig(os.path.join(THIS_DIR, "bo_resource.pdf"), bbox_inches="tight")
    plt.close(fr)
    for lig, oracle, tgt, stem in (("3nq9", ORACLE_3NQ9, "3NQ9", "bo_curve_3nq9"),
                                   ("4jsz", ORACLE_4JSZ, "4JSZ", "bo_curve_4jsz")):
        fc, ac = plt.subplots(figsize=(6.2, 5.3))
        panel_curve(ac, recov, lig, oracle, tgt, legend_in_title=False)
        fc.tight_layout()
        for _l in ac.get_xticklabels() + ac.get_yticklabels():
            _l.set_fontweight("bold")
        fc.savefig(os.path.join(THIS_DIR, f"{stem}.pdf"))
        plt.close(fc)

    print(f"[OK] {OUT_PNG}")
    print(f"[OK] {OUT_PDF}")
    print("[OK] + bo_resource.pdf, bo_curve_3nq9.pdf, bo_curve_4jsz.pdf")


if __name__ == "__main__":
    main()
