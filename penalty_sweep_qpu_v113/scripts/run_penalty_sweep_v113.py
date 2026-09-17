#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
DEFAULT_WORK_ROOT = ROOT / "qa_sweeps/1.penalty_test/_cases"
DEFAULT_SWEEP_ROOT = ROOT / "penalty_sweep_qpu_v113"
DEFAULT_PYTHON_SAMPLE = os.environ.get("QDOCK_PYTHON_SAMPLE", sys.executable)
DEFAULT_PYTHON_EVAL = os.environ.get("QDOCK_PYTHON_EVAL", sys.executable)


def frange_inclusive(start: float, stop: float, step: float):
    vals = []
    cur = start
    # Small tolerance to avoid floating-point miss at upper bound.
    while cur <= stop + 1e-9:
        vals.append(round(cur, 10))
        cur += step
    return vals


def format_k(value: float, decimals: int):
    return f"{value:.{decimals}f}"


def build_case_names(ligands, kdist_values, kmono_values):
    names = []
    for lig in ligands:
        for kd in kdist_values:
            for km in kmono_values:
                names.append(
                    f"{lig}_Kdist{format_k(kd, 1)}_Kmono{format_k(km, 1)}"
                )
    return names


def ensure_cases_exist(case_root: Path, case_names):
    missing = [name for name in case_names if not (case_root / name).is_dir()]
    return missing


def run_cmd(cmd, env, log_path: Path, progress_every: int = 50, append: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prog_re = re.compile(r"^\[(\d+)/(\d+)\]\s+(start|done|skip)\s+")
    mode = "a" if append else "w"
    with log_path.open(mode) as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            txt = line.rstrip()
            if not txt:
                continue
            if txt.startswith("run_tag=") or txt.startswith("ok_cases=") or txt.startswith("success_rate="):
                print(txt, flush=True)
                continue
            m = prog_re.match(txt)
            if not m:
                continue
            idx = int(m.group(1))
            total = int(m.group(2))
            if idx in (1, total) or idx % progress_every == 0:
                print(txt, flush=True)
        return proc.wait()


def load_json(path: Path):
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def load_total_reads(samples_json: Path):
    data = load_json(samples_json)
    if data is None:
        return 0
    if isinstance(data, dict):
        samples = data.get("samples") or []
        energies = data.get("energies") or []
        return max(len(samples), len(energies))
    if isinstance(data, list):
        return len(data)
    return 0


def compute_quality(report_csv: Path, n_reads_total: int):
    rows = []
    if report_csv.exists():
        with report_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

    if n_reads_total <= 0:
        return {
            "n_reads": 0,
            "n_valid": 0,
            "n_success_valid": 0,
            "validity_rate": 0.0,
            "quality": 0.0,
            "success_rate": None,
        }

    n = min(len(rows), n_reads_total) if rows else 0
    if n == 0:
        return {
            "n_reads": n_reads_total,
            "n_valid": 0,
            "n_success_valid": 0,
            "validity_rate": 0.0,
            "quality": 0.0,
            "success_rate": None,
        }

    n_valid = 0
    n_success_valid = 0
    for r in rows[:n]:
        feasible = str(r.get("Feasible", "")).strip().lower() == "true"
        success = str(r.get("Success(<=2.0A)", "")).strip().lower() == "true"
        if feasible:
            n_valid += 1
            if success:
                n_success_valid += 1

    validity_rate = n_valid / n_reads_total
    quality = n_success_valid / n_reads_total
    success_rate = (n_success_valid / n_valid) if n_valid > 0 else None
    return {
        "n_reads": n_reads_total,
        "n_valid": n_valid,
        "n_success_valid": n_success_valid,
        "validity_rate": validity_rate,
        "quality": quality,
        "success_rate": success_rate,
    }


def parse_case_name(case_name: str):
    m = re.match(
        r"^(?P<lig>[0-9a-z]+)_Kdist(?P<kd>[0-9.]+)_Kmono(?P<km>[0-9.]+)$",
        case_name,
    )
    if not m:
        return None
    return m.group("lig"), float(m.group("kd")), float(m.group("km"))


def collect_case_rows(sample_root: Path, case_names, run_tag: str):
    rows = []
    missing_dirs = []
    for case_name in case_names:
        parsed = parse_case_name(case_name)
        if parsed is None:
            continue
        lig, kd, km = parsed
        run_dir = sample_root / case_name / run_tag
        if not run_dir.is_dir():
            missing_dirs.append(case_name)
            continue

        report_csv = run_dir / f"{lig}_ligand_report.csv"
        samples_json = run_dir / f"{lig}_ligand_samples.json"
        result_json = run_dir / "result.json"
        meta_json = run_dir / f"{lig}_ligand_sampler_meta.json"

        n_reads_total = load_total_reads(samples_json)
        q = compute_quality(report_csv, n_reads_total=n_reads_total)

        result = load_json(result_json) or {}
        meta = load_json(meta_json) or {}
        timing = meta.get("timing", {}) if isinstance(meta, dict) else {}
        qpu_access_us = timing.get("qpu_access_time")
        qpu_sampling_us = timing.get("qpu_sampling_time")

        rows.append(
            {
                "ligand": lig,
                "case_name": case_name,
                "run_tag": run_tag,
                "Kdist": kd,
                "Kmono": km,
                "status": result.get("status"),
                "mRMSD": result.get("mRMSD"),
                "n_poses": result.get("n_poses"),
                "n_reads": q["n_reads"],
                "n_valid": q["n_valid"],
                "n_success_valid": q["n_success_valid"],
                "quality": q["quality"],
                "validity_rate": q["validity_rate"],
                "success_rate": q["success_rate"],
                "qpu_access_sec": (qpu_access_us / 1_000_000.0) if isinstance(qpu_access_us, (int, float)) else None,
                "qpu_sampling_sec": (qpu_sampling_us / 1_000_000.0) if isinstance(qpu_sampling_us, (int, float)) else None,
                "solver": meta.get("solver"),
                "physical_qubits": meta.get("physical_qubits"),
                "max_chain_length": meta.get("max_chain_length"),
                "report_csv": str(report_csv) if report_csv.exists() else "",
                "samples_json": str(samples_json) if samples_json.exists() else "",
                "result_json": str(result_json) if result_json.exists() else "",
            }
        )
    return rows, missing_dirs


def filter_pending_cases(sample_root: Path, case_names, run_tag: str):
    pending = []
    completed = []
    for case_name in case_names:
        parsed = parse_case_name(case_name)
        if parsed is None:
            continue
        lig, _, _ = parsed
        run_dir = sample_root / case_name / run_tag
        meta_ok = (run_dir / f"{lig}_ligand_sampler_meta.json").exists()
        samples_ok = (run_dir / f"{lig}_ligand_samples.json").exists()
        if meta_ok and samples_ok:
            completed.append(case_name)
        else:
            pending.append(case_name)
    return pending, completed


def mean_or_none(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def summarize_by_ligand(rows):
    by_lig = defaultdict(list)
    for r in rows:
        by_lig[r["ligand"]].append(r)

    out = []
    for lig, lig_rows in sorted(by_lig.items()):
        best = sorted(
            lig_rows,
            key=lambda r: (
                -(r["quality"] if r["quality"] is not None else -1),
                -(r["validity_rate"] if r["validity_rate"] is not None else -1),
                (r["mRMSD"] if isinstance(r["mRMSD"], (int, float)) else 1e9),
            ),
        )[0]
        out.append(
            {
                "ligand": lig,
                "n_cases": len(lig_rows),
                "quality_mean": mean_or_none([r["quality"] for r in lig_rows]),
                "quality_max": best.get("quality"),
                "best_Kdist": best.get("Kdist"),
                "best_Kmono": best.get("Kmono"),
                "best_validity_rate": best.get("validity_rate"),
                "best_mRMSD": best.get("mRMSD"),
                "qpu_access_sec_mean": mean_or_none([r["qpu_access_sec"] for r in lig_rows]),
            }
        )
    return out


def summarize_balanced(rows, tau: float = 0.25):
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[(r["Kdist"], r["Kmono"])][r["ligand"]] = r

    out = []
    for (kd, km), d in sorted(by_pair.items()):
        if "3nq9" not in d or "4jsz" not in d:
            continue
        q1 = d["3nq9"]["quality"]
        q2 = d["4jsz"]["quality"]
        v1 = d["3nq9"]["validity_rate"]
        v2 = d["4jsz"]["validity_rate"]
        if q1 is None or q2 is None:
            continue
        min_q = min(q1, q2)
        mean_q = (q1 + q2) / 2.0
        min_v = min(v1, v2) if (v1 is not None and v2 is not None) else None
        robust = None
        if min_v is not None:
            robust = min_q - 0.25 * max(0.0, tau - min_v)
        out.append(
            {
                "Kdist": kd,
                "Kmono": km,
                "quality_3nq9": q1,
                "quality_4jsz": q2,
                "validity_3nq9": v1,
                "validity_4jsz": v2,
                "balanced_min_quality": min_q,
                "balanced_mean_quality": mean_q,
                "balanced_min_validity": min_v,
                "robust_score_constrained": robust,
            }
        )
    return out


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_heatmap(rows, metric, out_path: Path, title: str, vmin=None, vmax=None, cmap="magma"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kdist_vals = sorted({float(r["Kdist"]) for r in rows})
    kmono_vals = sorted({float(r["Kmono"]) for r in rows})
    mat = np.full((len(kdist_vals), len(kmono_vals)), np.nan)

    kdist_idx = {v: i for i, v in enumerate(kdist_vals)}
    kmono_idx = {v: i for i, v in enumerate(kmono_vals)}
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        i = kdist_idx[float(r["Kdist"])]
        j = kmono_idx[float(r["Kmono"])]
        mat[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Kmono")
    ax.set_ylabel("Kdist")
    ax.set_xticks(range(len(kmono_vals)))
    ax.set_xticklabels([format_k(v, 1) for v in kmono_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(kdist_vals)))
    ax.set_yticklabels([format_k(v, 1) for v in kdist_vals])
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(path: Path, run_tag: str, solver: str, num_reads: int, rows, ligand_rows, balanced_rows):
    lines = []
    lines.append(f"# Penalty Sweep QPU ({run_tag})")
    lines.append("")
    lines.append(f"- Solver: `{solver}`")
    lines.append(f"- Num reads: `{num_reads}`")
    lines.append("- Grid: `Kdist=0.5..10.0 (step 0.5)`, `Kmono=1.0..20.0 (step 1.0)`")
    lines.append(f"- Collected rows: `{len(rows)}`")
    lines.append("")
    lines.append("## Best By Ligand (quality)")
    lines.append("")
    for r in ligand_rows:
        lines.append(
            f"- {r['ligand']}: quality_max={r['quality_max']:.6f} at "
            f"(Kdist,Kmono)=({r['best_Kdist']:.1f},{r['best_Kmono']:.1f}), "
            f"validity={r['best_validity_rate']:.6f}, mRMSD={r['best_mRMSD']}"
        )

    lines.append("")
    lines.append("## Best Balanced")
    lines.append("")
    if balanced_rows:
        best_min_q = sorted(
            balanced_rows,
            key=lambda r: r["balanced_min_quality"],
            reverse=True,
        )[0]
        lines.append(
            f"- balanced_min_quality best: {best_min_q['balanced_min_quality']:.6f} at "
            f"(Kdist,Kmono)=({best_min_q['Kdist']:.1f},{best_min_q['Kmono']:.1f})"
        )
        best_robust = sorted(
            [r for r in balanced_rows if r.get("robust_score_constrained") is not None],
            key=lambda r: r["robust_score_constrained"],
            reverse=True,
        )[0]
        lines.append(
            f"- robust_score_constrained best: {best_robust['robust_score_constrained']:.6f} at "
            f"(Kdist,Kmono)=({best_robust['Kdist']:.1f},{best_robust['Kmono']:.1f})"
        )
    else:
        lines.append("- No balanced rows available.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Run full penalty sweep on Advantage2_system and build heatmaps.")
    ap.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    ap.add_argument("--sweep-root", default=str(DEFAULT_SWEEP_ROOT))
    ap.add_argument("--python-sample", default=str(DEFAULT_PYTHON_SAMPLE))
    ap.add_argument("--python-eval", default=str(DEFAULT_PYTHON_EVAL))
    ap.add_argument("--solver", default="Advantage2_system1.13;graph_id=01e1ea5685")
    ap.add_argument("--num-reads", type=int, default=1000)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--skip-run", action="store_true", help="Skip sampling/eval and only aggregate existing run-tag.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip cases that already have >=1 completed sample run under sample root.",
    )
    args = ap.parse_args()

    work_root = Path(args.work_root)
    sweep_root = Path(args.sweep_root)
    config_dir = sweep_root / "configs"
    log_dir = sweep_root / "logs"
    result_dir = sweep_root / "results"
    fig_dir = sweep_root / "figures"
    report_dir = sweep_root / "report"
    sample_root = sweep_root / "samples"
    for d in [config_dir, log_dir, result_dir, fig_dir, report_dir, sample_root]:
        d.mkdir(parents=True, exist_ok=True)

    ligands = ["3nq9", "4jsz"]
    kdist_values = frange_inclusive(0.5, 10.0, 0.5)
    kmono_values = frange_inclusive(1.0, 20.0, 1.0)
    case_names = build_case_names(ligands, kdist_values, kmono_values)

    missing_cases = ensure_cases_exist(work_root, case_names)
    if missing_cases:
        raise SystemExit(f"Missing {len(missing_cases)} case dirs under {work_root}. Example: {missing_cases[:5]}")

    run_tag = args.run_tag or f"psweep112_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg_path = config_dir / f"sampler_{run_tag}.json"
    cfg = {
        "backend": "dwave_qpu",
        "backend_params": {"solver": args.solver},
        "sampler_kwargs": {
            "num_reads": int(args.num_reads),
            "return_embedding": True,
        },
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    cases_list_path = config_dir / f"cases_{run_tag}.txt"
    cases_list_path.write_text("\n".join(case_names) + "\n")
    run_case_names = list(case_names)
    completed_cases = []
    if args.resume:
        run_case_names, completed_cases = filter_pending_cases(sample_root, case_names, run_tag)
    run_cases_list_path = config_dir / f"cases_to_run_{run_tag}.txt"
    run_cases_list_path.write_text("\n".join(run_case_names) + "\n")

    if not args.skip_run and run_case_names:
        base_env = os.environ.copy()
        base_env["QDOCK_QPU_WORKDIR"] = str(work_root)
        base_env["QDOCK_SAMPLE_DIR"] = str(sample_root)
        base_env["QDOCK_EVAL_DIR"] = str(sample_root)
        base_env["QDOCK_PDB_IDS"] = ",".join(run_case_names)
        base_env["QDOCK_RUN_TAG"] = run_tag
        base_env["QDOCK_SAMPLER_CONFIG"] = str(cfg_path)
        # When resuming we already pass only pending case names via QDOCK_PDB_IDS.
        # Setting QDOCK_QPU_SKIP_RUNS_GTE here can incorrectly skip those pending
        # cases if previous runs exist under the same case root.
        base_env.pop("QDOCK_QPU_SKIP_RUNS_GTE", None)

        sample_log = log_dir / f"{run_tag}_sample.log"
        eval_log = log_dir / f"{run_tag}_eval.log"

        print(
            f"[sample] run_tag={run_tag} cases_total={len(case_names)} "
            f"cases_to_run={len(run_case_names)} resume={args.resume}",
            flush=True,
        )
        rc_sample = run_cmd(
            [args.python_sample, "D_qpu_sample_fam.py"],
            base_env,
            sample_log,
            append=args.resume,
        )
        if rc_sample != 0:
            raise SystemExit(f"Sampling failed with rc={rc_sample}. See {sample_log}")

        eval_env = dict(base_env)
        eval_env["QDOCK_REPORT"] = "1"
        print(f"[eval] run_tag={run_tag}", flush=True)
        rc_eval = run_cmd(
            [args.python_eval, "D_qpu_eval_fam.py"],
            eval_env,
            eval_log,
            append=args.resume,
        )
        if rc_eval != 0:
            raise SystemExit(f"Eval failed with rc={rc_eval}. See {eval_log}")

    rows, missing_dirs = collect_case_rows(sample_root, case_names, run_tag)
    if not rows:
        raise SystemExit(f"No rows collected under {sample_root} for run_tag={run_tag}")

    case_csv = result_dir / f"penalty_sweep_case_metrics_{run_tag}.csv"
    case_fields = [
        "ligand",
        "case_name",
        "run_tag",
        "Kdist",
        "Kmono",
        "status",
        "mRMSD",
        "n_poses",
        "n_reads",
        "n_valid",
        "n_success_valid",
        "quality",
        "validity_rate",
        "success_rate",
        "qpu_access_sec",
        "qpu_sampling_sec",
        "solver",
        "physical_qubits",
        "max_chain_length",
        "report_csv",
        "samples_json",
        "result_json",
    ]
    write_csv(case_csv, rows, case_fields)

    ligand_rows = summarize_by_ligand(rows)
    ligand_csv = result_dir / f"penalty_sweep_ligand_summary_{run_tag}.csv"
    ligand_fields = [
        "ligand",
        "n_cases",
        "quality_mean",
        "quality_max",
        "best_Kdist",
        "best_Kmono",
        "best_validity_rate",
        "best_mRMSD",
        "qpu_access_sec_mean",
    ]
    write_csv(ligand_csv, ligand_rows, ligand_fields)

    balanced_rows = summarize_balanced(rows, tau=0.25)
    balanced_rows = sorted(
        balanced_rows,
        key=lambda r: (r["balanced_min_quality"], r["robust_score_constrained"] if r["robust_score_constrained"] is not None else -1),
        reverse=True,
    )
    balanced_csv = result_dir / f"penalty_sweep_balanced_metrics_{run_tag}.csv"
    balanced_fields = [
        "Kdist",
        "Kmono",
        "quality_3nq9",
        "quality_4jsz",
        "validity_3nq9",
        "validity_4jsz",
        "balanced_min_quality",
        "balanced_mean_quality",
        "balanced_min_validity",
        "robust_score_constrained",
    ]
    write_csv(balanced_csv, balanced_rows, balanced_fields)

    by_lig = defaultdict(list)
    for r in rows:
        by_lig[r["ligand"]].append(r)
    for lig in sorted(by_lig):
        plot_heatmap(
            by_lig[lig],
            metric="quality",
            out_path=fig_dir / f"penalty_quality_heatmap_{lig}_{run_tag}.png",
            title=f"{lig} Quality (n_success_valid / n_reads)",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        plot_heatmap(
            by_lig[lig],
            metric="validity_rate",
            out_path=fig_dir / f"penalty_validity_heatmap_{lig}_{run_tag}.png",
            title=f"{lig} Validity Rate (n_valid / n_reads)",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )

    if balanced_rows:
        plot_heatmap(
            balanced_rows,
            metric="balanced_min_quality",
            out_path=fig_dir / f"penalty_balanced_min_quality_heatmap_{run_tag}.png",
            title="Balanced Min Quality across ligands",
            vmin=0.0,
            vmax=1.0,
            cmap="plasma",
        )

    report_path = report_dir / f"penalty_sweep_report_{run_tag}.md"
    write_report(report_path, run_tag, args.solver, int(args.num_reads), rows, ligand_rows, balanced_rows)

    manifest = {
        "run_tag": run_tag,
        "solver": args.solver,
        "num_reads": int(args.num_reads),
        "work_root": str(work_root),
        "sample_root": str(sample_root),
        "case_count_requested": len(case_names),
        "case_count_to_run": len(run_case_names),
        "case_count_already_completed": len(completed_cases),
        "case_count_collected": len(rows),
        "missing_case_output_dirs": len(missing_dirs),
        "missing_case_output_examples": missing_dirs[:20],
        "outputs": {
            "sampler_config": str(cfg_path),
            "cases_list": str(cases_list_path),
            "run_cases_list": str(run_cases_list_path),
            "case_csv": str(case_csv),
            "ligand_csv": str(ligand_csv),
            "balanced_csv": str(balanced_csv),
            "report_md": str(report_path),
            "figures_dir": str(fig_dir),
        },
    }
    manifest_path = result_dir / f"penalty_sweep_manifest_{run_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"run_tag={run_tag}")
    print(f"case_csv={case_csv}")
    print(f"ligand_csv={ligand_csv}")
    print(f"balanced_csv={balanced_csv}")
    print(f"report_md={report_path}")
    print(f"manifest_json={manifest_path}")


if __name__ == "__main__":
    main()
