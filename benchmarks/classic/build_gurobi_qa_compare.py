#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Resolve the repository root from this script unless explicitly overridden.
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("QDOCK_ROOT", str(DEFAULT_ROOT))).expanduser().resolve()

QUBO_DIR = ROOT / "qa_sweeps/1.penalty_test/_cases"
COMPARE_SAMPLE_DIR = ROOT / "compare_advantage"
COMPARE_SUMMARY = ROOT / "compare_advantage/eval/compare_advantage_run_summary_34col.csv"

# Gurobi outputs live under benchmarks (may be symlink); use script location if needed
SCRIPT_ROOT = Path(__file__).resolve().parents[2]
GUROBI_DIR = SCRIPT_ROOT / "classic/gurobi" if "benchmarks" in SCRIPT_ROOT.parts else ROOT / "benchmarks/classic/gurobi"

OUT_CSV = GUROBI_DIR / "qa_gurobi_compare.csv"

SOLVERS = ["Zephyr", "Pegasus"]


def load_qubo(path: Path) -> Dict[Tuple[str, str], float]:
    obj = np.load(path, allow_pickle=True)
    qubo = obj.item()
    return qubo


def filter_qubo(qubo: Dict[Tuple[str, str], float], vars_keep: set[str]) -> Dict[Tuple[str, str], float]:
    out = {}
    for (u, v), c in qubo.items():
        if u in vars_keep and v in vars_keep:
            out[(u, v)] = c
    return out


def energy_for_sample(qubo_terms, sample_dict) -> float:
    e = 0.0
    for (u, v), c in qubo_terms:
        e += c * sample_dict.get(u, 0) * sample_dict.get(v, 0)
    return e


def best_energy_from_samples(samples_path: Path, qubo_subset: Dict[Tuple[str, str], float]) -> float:
    data = json.load(open(samples_path))
    samples = data["samples"]
    terms = list(qubo_subset.items())
    best = None
    for s in samples:
        e = energy_for_sample(terms, s)
        if best is None or e < best:
            best = e
    return float(best) if best is not None else float("nan")


def main() -> None:
    if not COMPARE_SUMMARY.exists():
        raise SystemExit(f"Missing compare summary: {COMPARE_SUMMARY}")

    df = pd.read_csv(COMPARE_SUMMARY)

    rows = []

    for ligand in sorted(df["ligand"].unique()):
        case = f"{ligand}_Kdist1.5_Kmono18.0"
        qubo_path = QUBO_DIR / case / "QUBOs" / f"{ligand}_ligand.npy"
        qubo_full = load_qubo(qubo_path)

        gjson = GUROBI_DIR / f"{ligand}_gurobi.json"
        gvars = GUROBI_DIR / f"{ligand}_vars.txt"
        if not gjson.exists() or not gvars.exists():
            raise SystemExit(f"Missing Gurobi outputs for {ligand}: {gjson} / {gvars}")
        g = json.load(open(gjson))
        vars_keep = set(gvars.read_text().splitlines())
        qubo_subset = filter_qubo(qubo_full, vars_keep)

        row = {
            "ligand": ligand,
            "gurobi_vars": g.get("n_vars"),
            "gurobi_terms": g.get("n_terms"),
            "gurobi_obj": g.get("obj_val"),
            "gurobi_bound": g.get("obj_bound"),
            "gurobi_gap": g.get("best_gap"),
            "gurobi_runtime_sec": g.get("runtime_sec"),
        }

        for solver in SOLVERS:
            sub = df[(df["solver"] == solver) & (df["ligand"] == ligand)]
            best_e = None
            for _, r in sub.iterrows():
                run_tag = r["run_tag"]
                samples_path = COMPARE_SAMPLE_DIR / case / run_tag / "QUBOs" / f"{ligand}_ligand_samples.json"
                if not samples_path.exists():
                    continue
                e = best_energy_from_samples(samples_path, qubo_subset)
                if best_e is None or e < best_e:
                    best_e = e

            row[f"qa_best_energy_{solver}"] = best_e
            if best_e is not None and g.get("obj_val") is not None:
                row[f"energy_gap_{solver}"] = best_e - g.get("obj_val")
            else:
                row[f"energy_gap_{solver}"] = None

            summary_path = ROOT / f"compare_advantage/eval/{solver}/{solver}_summary_by_ligand.csv"
            if summary_path.exists():
                sm = pd.read_csv(summary_path)
                sm_row = sm[sm["ligand"] == ligand].iloc[0]
                row[f"quality_rate_{solver}"] = sm_row.get("quality_rate")
                row[f"validity_rate_{solver}"] = sm_row.get("validity_rate")
                row[f"tts_99_quality_sec_{solver}"] = sm_row.get("tts_99_quality_sec")
                row[f"tts_99_success_sec_{solver}"] = sm_row.get("tts_99_success_sec")

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
