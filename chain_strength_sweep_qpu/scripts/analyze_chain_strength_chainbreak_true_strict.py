#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_RE = re.compile(r"csopt1full_cbftrue_cs(?P<cs>[0-9p]+)_r(?P<rep>[0-9]+)_")
REPO_ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()


def parse_bool(v):
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_run_info(run_tag: str):
    m = RUN_RE.search(run_tag)
    if not m:
        return None
    cs = float(m.group("cs").replace("p", "."))
    rep = int(m.group("rep"))
    return cs, rep


def load_run_metrics(run_dir: Path, ligand: str):
    report_csv = run_dir / f"{ligand}_ligand_report.csv"
    meta_json = run_dir / f"{ligand}_ligand_sampler_meta.json"
    samples_json = run_dir / f"{ligand}_ligand_samples.json"
    if not report_csv.exists() or not meta_json.exists() or not samples_json.exists():
        raise FileNotFoundError(f"missing files in {run_dir} for {ligand}")

    with meta_json.open("r") as f:
        meta = json.load(f)
    with samples_json.open("r") as f:
        sj = json.load(f)

    timing = (meta.get("timing", {}) or {})
    num_reads = int(meta.get("num_reads", 0) or 0)
    if num_reads <= 0:
        num_reads = 1
    qpu_access_time_us = float(timing.get("qpu_access_time", 0.0))
    t_shot_sec = (qpu_access_time_us / 1e6) / num_reads

    n_rows = 0
    n_valid = 0
    n_success_valid = 0
    with report_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n_rows += 1
            valid = parse_bool(row.get("Feasible", ""))
            success = parse_bool(row.get("Success(<=2.0A)", ""))
            if valid:
                n_valid += 1
                if success:
                    n_success_valid += 1

    validity = n_valid / num_reads
    success_rate = (n_success_valid / n_valid) if n_valid > 0 else math.nan
    quality = n_success_valid / num_reads
    if quality <= 0:
        tts = math.inf
    elif quality >= 1:
        tts = t_shot_sec
    else:
        tts = t_shot_sec * math.log(0.01) / math.log(1.0 - quality)

    cbf = sj.get("chain_break_fraction")
    if isinstance(cbf, list) and cbf:
        cbf_mean = float(sum(float(x) for x in cbf) / len(cbf))
    else:
        cbf_mean = math.nan

    return {
        "n_reads": num_reads,
        "n_report_rows": n_rows,
        "n_valid": n_valid,
        "n_success_valid": n_success_valid,
        "quality": quality,
        "validity_rate": validity,
        "success_rate": success_rate,
        "t_shot_sec": t_shot_sec,
        "tts_0p99_sec": tts,
        "chain_break_fraction_mean": cbf_mean,
        "embedding_cached": parse_bool(meta.get("embedding_cached", False)),
    }


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def agg_rows(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["ligand"], r["chain_strength"])].append(r)
    out = []
    for (lig, cs), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        qs = [v["quality"] for v in vals]
        ts = [v["tts_0p99_sec"] for v in vals if math.isfinite(v["tts_0p99_sec"])]
        cbs = [v["chain_break_fraction_mean"] for v in vals if math.isfinite(v["chain_break_fraction_mean"])]
        out.append(
            {
                "ligand": lig,
                "chain_strength": cs,
                "n_repeats": len(vals),
                "quality_mean": sum(qs) / len(qs),
                "quality_min": min(qs),
                "quality_max": max(qs),
                "tts_0p99_mean_sec": (sum(ts) / len(ts)) if ts else "",
                "tts_0p99_min_sec": min(ts) if ts else "",
                "tts_0p99_max_sec": max(ts) if ts else "",
                "chain_break_fraction_mean": (sum(cbs) / len(cbs)) if cbs else "",
                "chain_break_fraction_min": min(cbs) if cbs else "",
                "chain_break_fraction_max": max(cbs) if cbs else "",
            }
        )
    return out


def make_boxplot(rows, metric_key, ylabel, title, out_file: Path, ylog=False):
    ligands = ["3nq9", "4jsz"]
    all_cs = sorted({float(r["chain_strength"]) for r in rows})
    labels = [f"{x:g}" for x in all_cs]
    by = {lig: defaultdict(list) for lig in ligands}
    for r in rows:
        by[r["ligand"]][float(r["chain_strength"])].append(float(r[metric_key]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=False)
    for i, lig in enumerate(ligands):
        ax = axes[i]
        data = []
        for cs in all_cs:
            vals = [v for v in by[lig][cs] if math.isfinite(v)]
            data.append(vals if vals else [math.nan])
        bp = ax.boxplot(data, tick_labels=labels, widths=0.65, patch_artist=True, showfliers=True)
        color = "#1f77b4" if lig == "3nq9" else "#ff7f0e"
        for b in bp["boxes"]:
            b.set_facecolor(color)
            b.set_alpha(0.33)
            b.set_edgecolor(color)
            b.set_linewidth(1.3)
        for m in bp["medians"]:
            m.set_color("#222222")
            m.set_linewidth(1.5)
        ax.set_title(lig)
        ax.set_xlabel("Chain Strength")
        ax.grid(axis="y", alpha=0.24)
        ax.tick_params(axis="x", rotation=45)
        if ylog:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep-root",
        default=str(
            REPO_ROOT
            / "chain_strength_sweep_qpu_fixedemb_v113_opt1_r1to10_fullcs_chainbreak_true"
        ),
    )
    args = ap.parse_args()

    root = Path(args.sweep_root)
    samples = root / "samples"
    results = root / "results"
    figures = root / "figures"
    case_lig = [
        ("3nq9_Kdist7.500_Kmono19.000", "3nq9"),
        ("4jsz_Kdist0.500_Kmono1.000", "4jsz"),
    ]

    rows = []
    missing = []
    for case, lig in case_lig:
        cdir = samples / case
        if not cdir.exists():
            continue
        for run_dir in sorted(cdir.iterdir()):
            if not run_dir.is_dir():
                continue
            info = parse_run_info(run_dir.name)
            if not info:
                continue
            cs, rep = info
            try:
                m = load_run_metrics(run_dir, lig)
            except Exception as e:
                missing.append((case, run_dir.name, str(e)))
                continue
            rows.append(
                {
                    "ligand": lig,
                    "case_name": case,
                    "chain_strength": cs,
                    "repeat": rep,
                    "run_tag": run_dir.name,
                    **m,
                }
            )

    raw = results / "chain_strength_fixedemb_opt1_r1to10_fullcs_chainbreak_true_quality_tts_raw.csv"
    agg = results / "chain_strength_fixedemb_opt1_r1to10_fullcs_chainbreak_true_quality_tts_agg.csv"
    miss = results / "chain_strength_fixedemb_opt1_r1to10_fullcs_chainbreak_true_missing_rows.csv"

    raw_fields = [
        "ligand",
        "case_name",
        "chain_strength",
        "repeat",
        "run_tag",
        "quality",
        "validity_rate",
        "success_rate",
        "tts_0p99_sec",
        "t_shot_sec",
        "chain_break_fraction_mean",
        "n_reads",
        "n_report_rows",
        "n_valid",
        "n_success_valid",
        "embedding_cached",
    ]
    write_csv(raw, rows, raw_fields)

    agg_data = agg_rows(rows)
    agg_fields = [
        "ligand",
        "chain_strength",
        "n_repeats",
        "quality_mean",
        "quality_min",
        "quality_max",
        "tts_0p99_mean_sec",
        "tts_0p99_min_sec",
        "tts_0p99_max_sec",
        "chain_break_fraction_mean",
        "chain_break_fraction_min",
        "chain_break_fraction_max",
    ]
    write_csv(agg, agg_data, agg_fields)

    make_boxplot(
        rows,
        metric_key="quality",
        ylabel="Quality",
        title="Chain-Strength Sweep (chain_break_fraction=true): Quality (r1-r10)",
        out_file=figures / "chain_strength_chainbreak_true_quality_boxplot_r1to10.png",
        ylog=False,
    )
    make_boxplot(
        rows,
        metric_key="tts_0p99_sec",
        ylabel="TTS_0.99 (sec)",
        title="Chain-Strength Sweep (chain_break_fraction=true): TTS_0.99 (r1-r10)",
        out_file=figures / "chain_strength_chainbreak_true_tts0p99_boxplot_r1to10.png",
        ylog=True,
    )
    make_boxplot(
        rows,
        metric_key="chain_break_fraction_mean",
        ylabel="Chain Break Fraction",
        title="Chain-Strength Sweep (chain_break_fraction=true): CBF (r1-r10)",
        out_file=figures / "chain_strength_chainbreak_true_fraction_boxplot_r1to10.png",
        ylog=False,
    )

    if missing:
        write_csv(miss, [{"case_name": a, "run_tag": b, "error": c} for a, b, c in missing], ["case_name", "run_tag", "error"])

    print(f"rows={len(rows)}")
    print(f"raw={raw}")
    print(f"agg={agg}")
    print(f"quality_fig={figures / 'chain_strength_chainbreak_true_quality_boxplot_r1to10.png'}")
    print(f"tts_fig={figures / 'chain_strength_chainbreak_true_tts0p99_boxplot_r1to10.png'}")
    print(f"cbf_fig={figures / 'chain_strength_chainbreak_true_fraction_boxplot_r1to10.png'}")
    if missing:
        print(f"missing_rows={len(missing)}")
        print(f"missing={miss}")


if __name__ == "__main__":
    main()
