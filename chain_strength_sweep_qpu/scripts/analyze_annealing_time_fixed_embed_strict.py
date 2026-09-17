#!/usr/bin/env python3
"""Recompute annealing-time sweep metrics with strict definitions.

Definitions (paper/table style):
- Validity   = #valid / N_reads
- Success    = #({valid AND RMSD<=2A}) / #valid (conditional)
- Quality    = #({valid AND RMSD<=2A}) / N_reads
- TTS_0.99   = t_shot * log(0.01) / log(1-quality), where
               t_shot = qpu_access_time_sec / N_reads
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import json
from collections import defaultdict


ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def safe_float(v, default=math.nan):
    try:
        return float(v)
    except Exception:
        return default


def read_manifest(path: Path):
    rows = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def compute_metrics(run_dir: Path, ligand: str):
    report_csv = run_dir / f"{ligand}_ligand_report.csv"
    meta_json = run_dir / f"{ligand}_ligand_sampler_meta.json"

    if not report_csv.exists() or not meta_json.exists():
        return None

    with meta_json.open() as f:
        meta = json.load(f)

    num_reads = safe_int(meta.get("num_reads", 0), 0)
    timing = meta.get("timing") or {}
    qpu_access_time_us = safe_float(timing.get("qpu_access_time", meta.get("qpu_access_time", math.nan)))

    n_valid = 0
    n_success_valid = 0
    n_rows = 0

    with report_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n_rows += 1
            feasible = parse_bool(row.get("Feasible", ""))
            success = parse_bool(row.get("Success(<=2.0A)", ""))
            if feasible:
                n_valid += 1
                if success:
                    n_success_valid += 1

    # fall back if num_reads is missing (should not happen in this sweep)
    if num_reads <= 0:
        num_reads = n_rows

    validity_rate = (n_valid / num_reads) if num_reads > 0 else math.nan
    success_rate = (n_success_valid / n_valid) if n_valid > 0 else math.nan
    quality = (n_success_valid / num_reads) if num_reads > 0 else math.nan

    if math.isfinite(qpu_access_time_us) and num_reads > 0:
        t_shot_sec = (qpu_access_time_us * 1e-6) / num_reads
    else:
        t_shot_sec = math.nan

    if not math.isfinite(quality):
        tts_099 = math.nan
    elif quality <= 0.0:
        tts_099 = math.inf
    elif quality >= 1.0:
        tts_099 = t_shot_sec
    else:
        tts_099 = t_shot_sec * math.log(0.01) / math.log(1.0 - quality)

    return {
        "n_reads": num_reads,
        "n_valid": n_valid,
        "n_success_valid": n_success_valid,
        "validity_rate": validity_rate,
        "success_rate": success_rate,
        "quality": quality,
        "qpu_access_time_us": qpu_access_time_us,
        "t_shot_sec": t_shot_sec,
        "tts099_sec": tts_099,
        "tts_0p99_sec": tts_099,
    }


def aggregate(rows):
    by = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok":
            key = (r["ligand"], safe_float(r["annealing_time"]))
            by[key].append(r)

    out = []
    for (lig, at), vals in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
        q = [safe_float(v["quality"]) for v in vals if math.isfinite(safe_float(v["quality"]))]
        v = [safe_float(vv["validity_rate"]) for vv in vals if math.isfinite(safe_float(vv["validity_rate"]))]
        t = [safe_float(vv["tts099_sec"]) for vv in vals if math.isfinite(safe_float(vv["tts099_sec"]))]

        def mean(xs):
            return sum(xs) / len(xs) if xs else math.nan

        def median(xs):
            if not xs:
                return math.nan
            ys = sorted(xs)
            m = len(ys) // 2
            if len(ys) % 2:
                return ys[m]
            return 0.5 * (ys[m - 1] + ys[m])

        def std(xs):
            if len(xs) < 2:
                return 0.0 if len(xs) == 1 else math.nan
            m = mean(xs)
            return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

        out.append(
            {
                "ligand": lig,
                "annealing_time": at,
                "n_total": len(vals),
                "n_ok": len(vals),
                "n_fail": 0,
                "quality_mean": mean(q),
                "quality_median": median(q),
                "quality_std": std(q),
                "validity_mean": mean(v),
                "success_rate_mean": mean([safe_float(vv["success_rate"]) for vv in vals if math.isfinite(safe_float(vv["success_rate"]))]),
                "tts099_mean_sec": mean(t),
                "tts099_median_sec": median(t),
                "tts099_std_sec": std(t),
                "tts_0p99_mean_sec": mean(t),
                "tts_0p99_median_sec": median(t),
                "tts_0p99_std_sec": std(t),
            }
        )
    return out


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def select_fields(rows, fields):
    out = []
    for r in rows:
        out.append({k: r.get(k, "") for k in fields})
    return out


def dedupe_ok_rows(rows):
    # keep last ok row by key so retried runs naturally override earlier rows
    out = {}
    order = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r.get("ligand", ""), safe_float(r.get("annealing_time")), safe_int(r.get("repeat", 0)))
        if key not in out:
            order.append(key)
        out[key] = r
    return [out[k] for k in order]


def analyze_manifest(manifest_rows):
    rows_ok = dedupe_ok_rows(manifest_rows)
    out = []
    for r in rows_ok:
        sample_log = r.get("sample_log", "")
        if not sample_log:
            continue
        run_tag = r.get("run_tag", "")
        if not run_tag:
            continue

        # sample_log lives under <sweep_root>/logs/<run_tag>_sample.log
        # run_dir is <sweep_root>/samples/<case_name>/<run_tag>
        sample_log_path = Path(sample_log)
        sweep_root = sample_log_path.parents[1]  # .../<sweep_root>
        run_dir = sweep_root / "samples" / r.get("case_name", "") / run_tag
        ligand = r.get("ligand", "")

        m = compute_metrics(run_dir, ligand)
        if m is None:
            continue

        out.append(
            {
                "ligand": ligand,
                "annealing_time": safe_float(r.get("annealing_time")),
                "repeat": safe_int(r.get("repeat", 0)),
                "status": "ok",
                **m,
            }
        )
    return out


def make_boxplots(raw_rows, fig_dir: Path):
    import matplotlib.pyplot as plt

    # exclude 1000 label per your current plotting rule
    rows = [r for r in raw_rows if safe_float(r["annealing_time"]) != 1000.0]

    ligands = sorted(set(r["ligand"] for r in rows))
    at_values = sorted(set(safe_float(r["annealing_time"]) for r in rows))

    def collect(metric):
        data = {lig: {at: [] for at in at_values} for lig in ligands}
        for r in rows:
            lig = r["ligand"]
            at = safe_float(r["annealing_time"])
            val = safe_float(r[metric])
            if math.isfinite(val):
                data[lig][at].append(val)
        return data

    qdata = collect("quality")
    tdata = collect("tts099_sec")

    fig_dir.mkdir(parents=True, exist_ok=True)

    # Quality
    fig, axes = plt.subplots(1, len(ligands), figsize=(6 * len(ligands), 5), squeeze=False)
    axes = axes[0]
    for i, lig in enumerate(ligands):
        ax = axes[i]
        ys = [qdata[lig][at] for at in at_values]
        ax.boxplot(ys, labels=[str(int(at)) if at.is_integer() else str(at) for at in at_values], showfliers=False)
        ax.set_title(f"{lig} Quality")
        ax.set_xlabel("annealing_time (us)")
        ax.set_ylabel("quality")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "anneal_time_quality_boxplot_no1000_with800.png", dpi=220)
    plt.close(fig)

    # TTS
    fig, axes = plt.subplots(1, len(ligands), figsize=(6 * len(ligands), 5), squeeze=False)
    axes = axes[0]
    for i, lig in enumerate(ligands):
        ax = axes[i]
        ys = [tdata[lig][at] for at in at_values]
        ax.boxplot(ys, labels=[str(int(at)) if at.is_integer() else str(at) for at in at_values], showfliers=False)
        ax.set_title(f"{lig} TTS_0.99")
        ax.set_xlabel("annealing_time (us)")
        ax.set_ylabel("TTS_0.99 (sec)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "anneal_time_tts099_boxplot_no1000_with800.png", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest-base",
        default=str(
            ROOT
            / "annealing_time_sweep_qpu_fixedemb_v113_opt1_r1to10/results/anneal_time_fixedemb_opt1_r1to10_manifest_20260407_182901.csv"
        ),
    )
    ap.add_argument(
        "--manifest-at800",
        default=str(
            ROOT
            / "annealing_time_sweep_qpu_fixedemb_v113_opt1_at800_r1to10/results/anneal_time_fixedemb_opt1_r1to10_manifest_20260407_205920.csv"
        ),
    )
    ap.add_argument(
        "--results-dir",
        default=str(
            ROOT
            / "annealing_time_sweep_qpu_fixedemb_v113_opt1_r1to10/results"
        ),
    )
    ap.add_argument(
        "--figures-dir",
        default=str(
            ROOT
            / "annealing_time_sweep_qpu_fixedemb_v113_opt1_r1to10/figures"
        ),
    )
    args = ap.parse_args()

    manifest_base = Path(args.manifest_base)
    manifest_at800 = Path(args.manifest_at800)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    rows_base_manifest = read_manifest(manifest_base)
    rows_base = analyze_manifest(rows_base_manifest)

    fields_raw = [
        "ligand",
        "annealing_time",
        "repeat",
        "status",
        "n_reads",
        "n_valid",
        "n_success_valid",
        "validity_rate",
        "success_rate",
        "quality",
        "qpu_access_time_us",
        "t_shot_sec",
        "tts099_sec",
        "tts_0p99_sec",
    ]

    write_csv(results_dir / "anneal_time_fixedemb_opt1_r1to10_quality_tts_raw.csv", rows_base, fields_raw)

    agg_base = aggregate(rows_base)
    fields_agg = [
        "ligand",
        "annealing_time",
        "n_total",
        "n_ok",
        "n_fail",
        "quality_mean",
        "quality_median",
        "quality_std",
        "validity_mean",
        "success_rate_mean",
        "tts099_mean_sec",
        "tts099_median_sec",
        "tts099_std_sec",
        "tts_0p99_mean_sec",
        "tts_0p99_median_sec",
        "tts_0p99_std_sec",
    ]
    write_csv(results_dir / "anneal_time_fixedemb_opt1_r1to10_quality_tts_agg.csv", agg_base, fields_agg)

    # Combine base + at800
    rows_combo_manifest = list(rows_base_manifest)
    if manifest_at800.exists():
        rows_combo_manifest.extend(read_manifest(manifest_at800))

    rows_combo = analyze_manifest(rows_combo_manifest)
    write_csv(results_dir / "anneal_time_fixedemb_opt1_r1to10_plus800_no1000_raw.csv", rows_combo, fields_raw)

    agg_combo = aggregate(rows_combo)
    write_csv(results_dir / "anneal_time_fixedemb_opt1_r1to10_plus800_quality_tts_agg.csv", agg_combo, fields_agg)

    rows_no1000 = [r for r in rows_combo if safe_float(r["annealing_time"]) != 1000.0]
    agg_no1000 = aggregate(rows_no1000)

    fields_agg_no1000 = [
        "ligand",
        "annealing_time",
        "n_ok",
        "quality_mean",
        "quality_std",
        "validity_mean",
        "success_rate_mean",
        "tts099_mean_sec",
        "tts099_std_sec",
        "tts_0p99_mean_sec",
        "tts_0p99_std_sec",
    ]
    write_csv(
        results_dir / "anneal_time_fixedemb_opt1_r1to10_plus800_no1000_agg.csv",
        select_fields(agg_no1000, fields_agg_no1000),
        fields_agg_no1000,
    )

    make_boxplots(rows_combo, figures_dir)

    print("Wrote:")
    print("-", results_dir / "anneal_time_fixedemb_opt1_r1to10_quality_tts_raw.csv")
    print("-", results_dir / "anneal_time_fixedemb_opt1_r1to10_quality_tts_agg.csv")
    print("-", results_dir / "anneal_time_fixedemb_opt1_r1to10_plus800_no1000_raw.csv")
    print("-", results_dir / "anneal_time_fixedemb_opt1_r1to10_plus800_quality_tts_agg.csv")
    print("-", results_dir / "anneal_time_fixedemb_opt1_r1to10_plus800_no1000_agg.csv")
    print("-", figures_dir / "anneal_time_quality_boxplot_no1000_with800.png")
    print("-", figures_dir / "anneal_time_tts099_boxplot_no1000_with800.png")


if __name__ == "__main__":
    main()
