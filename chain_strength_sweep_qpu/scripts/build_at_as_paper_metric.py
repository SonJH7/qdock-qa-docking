#!/usr/bin/env python3
"""Build the retained anneal-time paper metrics.

Anneal-shape metrics are maintained separately by the six-shape ``as_jjin``
campaign and its dedicated analysis/plotting scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()

# AT sources
AT3_MANIFEST = ROOT / "annealing_time_sweep_qpu_fixedemb_v113_opt1_3nq9_cs1p0_upto800_r1to10/results/anneal_time_fixedemb_opt1_3nq9_cs1p0_upto800_r1to10_manifest_20260408_023341.csv"
AT2_MANIFEST = ROOT / "annealing_time_sweep_qpu_fixedemb_v113_opt1_r1to10/results/anneal_time_fixedemb_opt1_r1to10_manifest_20260407_182901.csv"
AT2_800_MANIFEST = ROOT / "annealing_time_sweep_qpu_fixedemb_v113_opt1_at800_r1to10/results/anneal_time_fixedemb_opt1_r1to10_manifest_20260407_205920.csv"

def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def safe_float(v, default=math.nan):
    try:
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        s = str(v).strip()
        if s == "":
            return default
        return int(float(v))
    except Exception:
        return default


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def derive_run_dir(row: dict) -> Path:
    sample_log = row.get("sample_log", "")
    case_name = row.get("case_name", "")
    run_tag = row.get("run_tag", "")
    if not sample_log or not case_name or not run_tag:
        raise RuntimeError(f"manifest row missing path fields: case={case_name}, run={run_tag}")
    sample_log_path = Path(sample_log)
    sweep_root = sample_log_path.parents[1]
    return sweep_root / "samples" / case_name / run_tag


def compute_quality_tts(run_dir: Path, ligand: str) -> dict:
    report_csv = run_dir / f"{ligand}_ligand_report.csv"
    meta_json = run_dir / f"{ligand}_ligand_sampler_meta.json"
    result_json = run_dir / "result.json"

    if not meta_json.exists():
        raise FileNotFoundError(f"missing sampler_meta: {meta_json}")

    meta = json.loads(meta_json.read_text())
    timing = meta.get("timing", {}) or {}
    num_reads = safe_int(meta.get("num_reads", 0), 0)
    if num_reads <= 0:
        raise RuntimeError(f"invalid num_reads in meta: {meta_json}")

    qpu_access_time_us = safe_float(timing.get("qpu_access_time", math.nan), math.nan)
    t_shot_sec = (
        (qpu_access_time_us * 1e-6) / num_reads if math.isfinite(qpu_access_time_us) and num_reads > 0 else math.nan
    )

    n_report_rows = 0
    n_valid = 0
    n_success_valid = 0

    if report_csv.exists():
        with report_csv.open(newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                n_report_rows += 1
                is_valid = parse_bool(r.get("Feasible", ""))
                is_success = parse_bool(r.get("Success(<=2.0A)", ""))
                if is_valid:
                    n_valid += 1
                    if is_success:
                        n_success_valid += 1
    else:
        # strict guard: report missing is allowed only when no poses were built
        if result_json.exists():
            rj = json.loads(result_json.read_text())
            if safe_int(rj.get("n_poses", 0), 0) > 0:
                raise RuntimeError(f"report missing but n_poses>0: {result_json}")

    validity_rate = n_valid / num_reads
    success_rate = (n_success_valid / n_valid) if n_valid > 0 else math.nan
    quality = n_success_valid / num_reads

    if quality <= 0:
        tts_099 = math.inf
    elif quality >= 1:
        tts_099 = t_shot_sec
    else:
        tts_099 = t_shot_sec * math.log(0.01) / math.log(1.0 - quality)

    return {
        "n_reads": num_reads,
        "n_report_rows": n_report_rows,
        "n_valid": n_valid,
        "n_success_valid": n_success_valid,
        "validity_rate": validity_rate,
        "success_rate": success_rate,
        "quality": quality,
        "qpu_access_time_us": qpu_access_time_us,
        "tts_0p99_sec": tts_099,
        "embedding_cached": bool(meta.get("embedding_cached", False)),
    }


def dedupe_rows(rows: list[dict], key_fields: list[str]) -> list[dict]:
    by_key = {}
    order = []
    for r in rows:
        key = tuple(r.get(k, "") for k in key_fields)
        if key not in by_key:
            order.append(key)
        by_key[key] = r
    return [by_key[k] for k in order]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def agg_at(rows: list[dict]) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        g[(r["ligand"], safe_float(r["annealing_time"]))].append(r)

    out = []
    for (lig, at), vals in sorted(g.items(), key=lambda x: (x[0][0], x[0][1])):
        qs = [safe_float(v["quality"]) for v in vals if math.isfinite(safe_float(v["quality"]))]
        ts = [safe_float(v["tts_0p99_sec"]) for v in vals if math.isfinite(safe_float(v["tts_0p99_sec"]))]
        out.append(
            {
                "ligand": lig,
                "annealing_time": at,
                "n_repeats": len(vals),
                "quality_mean": (sum(qs) / len(qs)) if qs else math.nan,
                "quality_min": min(qs) if qs else math.nan,
                "quality_max": max(qs) if qs else math.nan,
                "tts_0p99_mean_sec": (sum(ts) / len(ts)) if ts else math.nan,
                "tts_0p99_min_sec": min(ts) if ts else math.nan,
                "tts_0p99_max_sec": max(ts) if ts else math.nan,
            }
        )
    return out


def plot_at(rows: list[dict], out_quality: Path, out_tts: Path) -> None:
    ligands = ["3nq9", "4jsz"]
    at_values = sorted({safe_float(r["annealing_time"]) for r in rows if math.isfinite(safe_float(r["annealing_time"]))})

    def collect(metric: str):
        data = {lig: {at: [] for at in at_values} for lig in ligands}
        for r in rows:
            lig = r["ligand"]
            if lig not in data:
                continue
            at = safe_float(r["annealing_time"])
            v = safe_float(r[metric])
            if math.isfinite(at) and math.isfinite(v):
                data[lig][at].append(v)
        return data

    qdata = collect("quality")
    tdata = collect("tts_0p99_sec")
    labels = [str(int(x)) if float(x).is_integer() else str(x) for x in at_values]

    out_quality.parent.mkdir(parents=True, exist_ok=True)
    out_tts.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    axes = axes[0]
    for i, lig in enumerate(ligands):
        ax = axes[i]
        ys = [qdata[lig][at] if qdata[lig][at] else [math.nan] for at in at_values]
        ax.boxplot(ys, tick_labels=labels, showfliers=False)
        ax.set_title(f"{lig} (cs={1.0 if lig=='3nq9' else 0.75})")
        ax.set_xlabel("annealing_time (us)")
        ax.set_ylabel("quality")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("AT Sweep (paper metric: quality=#success_valid/n_reads)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_quality, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    axes = axes[0]
    for i, lig in enumerate(ligands):
        ax = axes[i]
        ys = [tdata[lig][at] if tdata[lig][at] else [math.nan] for at in at_values]
        ax.boxplot(ys, tick_labels=labels, showfliers=False)
        ax.set_title(f"{lig} (cs={1.0 if lig=='3nq9' else 0.75})")
        ax.set_xlabel("annealing_time (us)")
        ax.set_ylabel("TTS_0.99 (sec)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("AT Sweep (paper metric)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_tts, dpi=300)
    plt.close(fig)


def build_at_rows() -> tuple[list[dict], list[dict]]:
    at3 = read_manifest(AT3_MANIFEST)
    at2 = read_manifest(AT2_MANIFEST)
    at2_800 = read_manifest(AT2_800_MANIFEST)

    # 3nq9: cs=1.0 / 4jsz: cs=0.75
    sel = []
    sel.extend(
        r
        for r in at3
        if r.get("status") == "ok"
        and r.get("ligand") == "3nq9"
        and abs(safe_float(r.get("chain_strength"), math.nan) - 1.0) < 1e-9
    )
    sel.extend(
        r
        for r in at2
        if r.get("status") == "ok"
        and r.get("ligand") == "4jsz"
        and abs(safe_float(r.get("chain_strength"), math.nan) - 0.75) < 1e-9
    )
    # supplement 4jsz@800 (and possible retried rows) from dedicated at800 rerun
    sel.extend(
        r
        for r in at2_800
        if r.get("status") == "ok"
        and r.get("ligand") == "4jsz"
        and abs(safe_float(r.get("chain_strength"), math.nan) - 0.75) < 1e-9
    )

    sel = dedupe_rows(sel, key_fields=["ligand", "annealing_time", "repeat"])
    out = []
    missing = []
    for r in sel:
        try:
            run_dir = derive_run_dir(r)
            m = compute_quality_tts(run_dir, r["ligand"])
            out.append(
                {
                    "ligand": r["ligand"],
                    "annealing_time": safe_float(r["annealing_time"]),
                    "repeat": safe_int(r["repeat"]),
                    "chain_strength": safe_float(r["chain_strength"]),
                    "run_tag": r["run_tag"],
                    "case_name": r.get("case_name", ""),
                    **m,
                }
            )
        except Exception as e:
            missing.append(
                {
                    "ligand": r.get("ligand", ""),
                    "annealing_time": r.get("annealing_time", ""),
                    "repeat": r.get("repeat", ""),
                    "run_tag": r.get("run_tag", ""),
                    "error": str(e),
                }
            )
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-at", default=str(ROOT / "at"))
    args = ap.parse_args()

    out_at = Path(args.out_at)

    # AT
    at_rows, at_missing = build_at_rows()
    at_results = out_at / "results"
    at_figures = out_at / "figures"
    write_csv(
        at_results / "at_quality_tts_raw.csv",
        at_rows,
        [
            "ligand",
            "annealing_time",
            "repeat",
            "chain_strength",
            "run_tag",
            "case_name",
            "n_reads",
            "n_report_rows",
            "n_valid",
            "n_success_valid",
            "validity_rate",
            "success_rate",
            "quality",
            "qpu_access_time_us",
            "tts_0p99_sec",
            "embedding_cached",
        ],
    )
    write_csv(
        at_results / "at_quality_tts_agg.csv",
        agg_at(at_rows),
        [
            "ligand",
            "annealing_time",
            "n_repeats",
            "quality_mean",
            "quality_min",
            "quality_max",
            "tts_0p99_mean_sec",
            "tts_0p99_min_sec",
            "tts_0p99_max_sec",
        ],
    )
    if at_missing:
        write_csv(
            at_results / "at_missing_rows.csv",
            at_missing,
            ["ligand", "annealing_time", "repeat", "run_tag", "error"],
        )
    plot_at(
        at_rows,
        out_quality=at_figures / "at_quality_boxplot.png",
        out_tts=at_figures / "at_tts0p99_boxplot.png",
    )

    print("Wrote:")
    print("-", at_results / "at_quality_tts_raw.csv")
    print("-", at_results / "at_quality_tts_agg.csv")
    if at_missing:
        print("-", at_results / "at_missing_rows.csv")
    print("-", at_figures / "at_quality_boxplot.png")
    print("-", at_figures / "at_tts0p99_boxplot.png")


if __name__ == "__main__":
    main()
