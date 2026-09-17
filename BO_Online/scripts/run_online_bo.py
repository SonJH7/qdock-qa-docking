#!/usr/bin/env python3
"""Online BO for penalty tuning with quality-centric objectives.

This script supports:
- qpu mode: sequentially queries QPU by generating QUBOs and evaluating samples.
- oracle mode: uses a precomputed table for smoke tests and local debugging.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm


def default_kdist_values() -> list[float]:
    return [0.1] + [0.5 * i for i in range(1, 21)]


def default_kmono_values() -> list[float]:
    return [0.1] + [float(i) for i in range(1, 21)]


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out or None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Online BO for QPU penalty tuning")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--out-dir", type=Path, default=repo_root / "BO_Online")
    parser.add_argument("--mode", choices=["qpu", "oracle"], default="qpu")
    parser.add_argument(
        "--oracle-csv",
        type=Path,
        default=repo_root / "BO" / "data" / "qpu_penalty_validity_summary.csv",
    )
    parser.add_argument(
        "--objective",
        choices=[
            "quality_min",
            "quality_mean",
            "quality_3nq9",
            "quality_4jsz",
            "robust_score_constrained",
        ],
        default="quality_min",
        help="Primary online objective; quality_min is the manuscript-aligned robust quality target.",
    )
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260316)
    parser.add_argument("--xi", type=float, default=0.02)
    parser.add_argument(
        "--policy",
        choices=["bo", "random"],
        default="bo",
        help="Candidate selection policy after init: BO acquisition or uniform random without replacement.",
    )
    parser.add_argument("--pdb-ids", type=str, default="3nq9,4jsz")
    parser.add_argument("--kdist-list", type=str, default=None)
    parser.add_argument("--kmono-list", type=str, default=None)
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--embed-reuse", action="store_true")
    parser.add_argument(
        "--sampler-config",
        type=Path,
        default=repo_root / "sampler_config_qpu_reads1000.json",
    )
    parser.add_argument("--plot-dpi", type=int, default=220)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing partial outputs in --out-dir/work/eval",
    )
    return parser.parse_args()


def pair_key(kdist: float, kmono: float) -> tuple[float, float]:
    return (round(float(kdist), 6), round(float(kmono), 6))


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    if np.issubdtype(values.dtype, np.number):
        return values.fillna(0).astype(float) > 0
    text = values.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


def compute_metrics_from_report(report_csv: Path, n_reads: int) -> dict[str, Any]:
    if not report_csv.exists():
        raise FileNotFoundError(f"report csv not found: {report_csv}")
    df = pd.read_csv(report_csv)
    feasible_col = None
    success_col = None
    for c in df.columns:
        cl = c.lower()
        if feasible_col is None and "feasible" in cl:
            feasible_col = c
        if success_col is None and "success" in cl:
            success_col = c
    if feasible_col is None or success_col is None:
        raise RuntimeError(f"required columns missing in {report_csv}")

    feasible = bool_series(df[feasible_col])
    success = bool_series(df[success_col])

    n_valid = int(feasible.sum())
    n_success_valid = int((feasible & success).sum())

    validity_rate = float(n_valid / n_reads) if n_reads > 0 else math.nan
    success_rate = float(n_success_valid / n_valid) if n_valid > 0 else math.nan
    quality = float(n_success_valid / n_reads) if n_reads > 0 else math.nan

    out: dict[str, Any] = {
        "n_report_rows": int(len(df)),
        "n_reads": int(n_reads),
        "n_valid": n_valid,
        "n_success_valid": n_success_valid,
        "validity_rate": validity_rate,
        "success_rate": success_rate,
        "quality": quality,
    }
    if "RMSD(A)" in df.columns:
        out["mRMSD_report"] = float(pd.to_numeric(df["RMSD(A)"], errors="coerce").min())
        out["avg_RMSD_report"] = float(pd.to_numeric(df["RMSD(A)"], errors="coerce").mean())
    return out


def resolve_report_csv_path(eval_result: dict[str, Any]) -> Path | None:
    paths = eval_result.get("paths", {}) or {}
    raw = paths.get("report_csv")
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
    out_dir = paths.get("output_dir")
    if out_dir:
        root = Path(out_dir)
        if root.is_dir():
            hits = sorted(root.glob("*_report.csv"))
            if hits:
                return hits[0]
    return None


def compute_metrics_without_report(n_reads: int, eval_result: dict[str, Any]) -> dict[str, Any]:
    # Fallback path: evaluator can return status=ok with n_poses=0, where no report csv is emitted.
    n_valid = int(eval_result.get("n_poses") or 0)
    n_success_valid = 0
    validity_rate = float(n_valid / n_reads) if n_reads > 0 else math.nan
    success_rate = float(n_success_valid / n_valid) if n_valid > 0 else math.nan
    quality = float(n_success_valid / n_reads) if n_reads > 0 else math.nan
    return {
        "n_report_rows": 0,
        "n_reads": int(n_reads),
        "n_valid": n_valid,
        "n_success_valid": n_success_valid,
        "validity_rate": validity_rate,
        "success_rate": success_rate,
        "quality": quality,
        "avg_RMSD_report": math.nan,
    }


class OracleEvaluator:
    def __init__(self, csv_path: Path, ligands: list[str], num_reads: int):
        self.csv_path = csv_path
        self.ligands = [x.lower() for x in ligands]
        self.num_reads = int(num_reads)

        df = pd.read_csv(csv_path)
        df["ligand"] = df["ligand"].str.lower()
        if set(self.ligands) - set(df["ligand"].unique()):
            raise ValueError("Oracle CSV is missing required ligands")

        needed = [
            "kdist",
            "kmono",
            "ligand",
            "quality",
            "validity_rate",
            "qpu_access_time",
            "qpu_sampling_time",
            "n_reads",
            "n_valid",
            "n_success_valid",
            "success_rate",
            "run_tag",
        ]
        for col in needed:
            if col not in df.columns:
                if col in {"n_reads", "n_valid", "n_success_valid", "success_rate", "run_tag", "qpu_sampling_time"}:
                    df[col] = np.nan
                else:
                    raise ValueError(f"Missing required column in oracle CSV: {col}")

        agg = (
            df.groupby(["kdist", "kmono", "ligand"], as_index=False)
            .agg(
                quality=("quality", "mean"),
                validity_rate=("validity_rate", "mean"),
                qpu_access_time=("qpu_access_time", "mean"),
                qpu_sampling_time=("qpu_sampling_time", "mean"),
                n_reads=("n_reads", "mean"),
                n_valid=("n_valid", "mean"),
                n_success_valid=("n_success_valid", "mean"),
                success_rate=("success_rate", "mean"),
                run_tag=("run_tag", "first"),
            )
            .reset_index(drop=True)
        )
        agg["k"] = agg.apply(lambda r: (pair_key(r["kdist"], r["kmono"]), r["ligand"]), axis=1)
        self.table = {k: row for k, row in zip(agg["k"], agg.to_dict(orient="records"))}

    def evaluate(self, kdist: float, kmono: float, query_tag: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        key = pair_key(kdist, kmono)
        for ligand in self.ligands:
            row = self.table.get((key, ligand))
            if row is None:
                raise KeyError(f"No oracle row for ligand={ligand}, k={key}")
            n_reads = int(round(row["n_reads"])) if np.isfinite(row["n_reads"]) else self.num_reads
            n_valid = int(round(row["n_valid"])) if np.isfinite(row["n_valid"]) else int(round(row["validity_rate"] * n_reads))
            n_success_valid = (
                int(round(row["n_success_valid"]))
                if np.isfinite(row["n_success_valid"])
                else int(round(row["quality"] * n_reads))
            )
            success_rate = (
                float(row["success_rate"])
                if np.isfinite(row["success_rate"])
                else (float(n_success_valid / n_valid) if n_valid > 0 else math.nan)
            )

            out.append(
                {
                    "mode": "oracle",
                    "query_tag": query_tag,
                    "ligand": ligand,
                    "kdist": float(kdist),
                    "kmono": float(kmono),
                    "n_reads": n_reads,
                    "n_valid": n_valid,
                    "n_success_valid": n_success_valid,
                    "validity_rate": float(row["validity_rate"]),
                    "success_rate": success_rate,
                    "quality": float(row["quality"]),
                    "qpu_access_time": float(row["qpu_access_time"]) if np.isfinite(row["qpu_access_time"]) else math.nan,
                    "qpu_sampling_time": float(row["qpu_sampling_time"]) if np.isfinite(row["qpu_sampling_time"]) else math.nan,
                    "sample_elapsed_sec": math.nan,
                    "mRMSD": math.nan,
                    "avg_RMSD": math.nan,
                    "physical_qubits": math.nan,
                    "max_chain_length": math.nan,
                    "raw_case_id": row.get("run_tag") or "oracle",
                }
            )
        return out


class QPUEvaluator:
    def __init__(
        self,
        repo_root: Path,
        out_dir: Path,
        ligands: list[str],
        num_reads: int,
        sampler_config: Path,
        embed_reuse: bool,
    ):
        self.repo_root = repo_root
        self.out_dir = out_dir
        self.ligands = [x.lower() for x in ligands]
        self.num_reads = int(num_reads)
        self.sampler_config = sampler_config
        self.embed_reuse = bool(embed_reuse)

        sys.path.insert(0, str(repo_root))

        import D_qpu_eval_fam as eval_mod
        import D_qpu_sample_fam as sample_mod
        import run_penalty_sweep_fam as penalty_mod

        self.eval_mod = eval_mod
        self.sample_mod = sample_mod
        self.penalty_mod = penalty_mod

        adfr_bin = os.environ.get("QDOCK_ADFR_BIN")
        if adfr_bin and Path(adfr_bin).exists():
            os.environ["PATH"] = adfr_bin + os.pathsep + os.environ.get("PATH", "")

        self.work_root = out_dir / "work"
        self.preproc_root = self.work_root / "_preproc"
        self.cases_root = self.work_root / "cases"
        self.sample_root = self.work_root / "samples"
        self.eval_root = self.work_root / "eval"
        self.preproc_root.mkdir(parents=True, exist_ok=True)
        self.cases_root.mkdir(parents=True, exist_ok=True)
        self.sample_root.mkdir(parents=True, exist_ok=True)
        self.eval_root.mkdir(parents=True, exist_ok=True)

        if not sampler_config.exists():
            raise FileNotFoundError(f"sampler config not found: {sampler_config}")
        os.environ["QDOCK_SAMPLER_CONFIG"] = str(sampler_config)

        try:
            self.sampler, self.sampler_kwargs, _, _ = self.sample_mod.load_sampler_from_env()
        except ModuleNotFoundError as exc:
            if "dwave" in str(exc):
                raise RuntimeError(
                    "QPU backend dependencies are missing. Install Ocean SDK packages "
                    "(e.g., dwave-system) in the active environment."
                ) from exc
            raise
        if "num_reads" not in self.sampler_kwargs:
            self.sampler_kwargs["num_reads"] = self.num_reads

        self.base_params = {"edge_cutoff": 1.87, "grid_length": 1.0}
        self.sample_params = {"n_pos": self.num_reads}
        self.empty_gasteiger: dict[str, Any] = {}

        self.prep: dict[str, dict[str, Any]] = {}
        for ligand in self.ligands:
            prep, err = self.penalty_mod.setup_preproc(
                ligand,
                str(self.repo_root),
                str(self.preproc_root),
                self.base_params,
            )
            if err:
                raise RuntimeError(f"Preprocessing failed for {ligand}: {err}")
            self.prep[ligand] = prep

    def evaluate(self, kdist: float, kmono: float, query_tag: str) -> list[dict[str, Any]]:
        os.environ["QDOCK_EMBEDDING_REUSE"] = "1" if self.embed_reuse else "0"

        out: list[dict[str, Any]] = []
        for ligand in self.ligands:
            prep = self.prep[ligand]
            case_name = f"{ligand}_Kdist{kdist:.3f}_Kmono{kmono:.3f}"
            case_dir = self.cases_root / case_name
            case_dir.mkdir(parents=True, exist_ok=True)

            params = {
                "edge_cutoff": self.base_params["edge_cutoff"],
                "grid_length": self.base_params["grid_length"],
                "K_dist": float(kdist),
                "K_mono": float(kmono),
            }

            qubo_path = self.penalty_mod.run_case_combo(
                prep["fam"],
                prep["preproc_dir"],
                prep["ligand_name"],
                str(case_dir),
                params,
            )
            self.penalty_mod.write_case_meta(
                str(case_dir),
                {
                    "ligand": ligand,
                    "case_name": case_name,
                    "K_dist": float(kdist),
                    "K_mono": float(kmono),
                    "edge_cutoff": self.base_params["edge_cutoff"],
                    "grid_length": self.base_params["grid_length"],
                    "qubo_path": qubo_path,
                    "query_tag": query_tag,
                },
            )

            sample_result = self.sample_mod.run_case(
                str(case_dir),
                str(self.sample_root),
                query_tag,
                self.sample_params,
                self.sampler,
                self.sampler_kwargs,
            )
            if sample_result.get("status") != "ok":
                raise RuntimeError(
                    f"Sampling failed for {case_name}: {sample_result.get('error_type')} {sample_result.get('error_msg')}"
                )

            eval_result = self.eval_mod.run_case(
                str(case_dir),
                str(self.sample_root),
                str(self.eval_root),
                {},
                self.empty_gasteiger,
                report=True,
                run_tag=query_tag,
                output_tag=query_tag,
            )
            if eval_result.get("status") != "ok":
                raise RuntimeError(
                    f"Evaluation failed for {case_name}: {eval_result.get('error_type')} {eval_result.get('error_msg')}"
                )

            n_reads = int(sample_result.get("sampler_meta", {}).get("num_reads") or self.num_reads)
            report_csv = resolve_report_csv_path(eval_result)
            if report_csv is not None:
                try:
                    metric = compute_metrics_from_report(report_csv, n_reads=n_reads)
                except Exception:
                    metric = compute_metrics_without_report(n_reads=n_reads, eval_result=eval_result)
            else:
                metric = compute_metrics_without_report(n_reads=n_reads, eval_result=eval_result)

            timing = sample_result.get("sampler_meta", {}).get("timing", {}) or {}
            qpu_access_time = timing.get("qpu_access_time")
            qpu_sampling_time = timing.get("qpu_sampling_time")

            out.append(
                {
                    "mode": "qpu",
                    "query_tag": query_tag,
                    "ligand": ligand,
                    "kdist": float(kdist),
                    "kmono": float(kmono),
                    "n_reads": int(metric["n_reads"]),
                    "n_valid": int(metric["n_valid"]),
                    "n_success_valid": int(metric["n_success_valid"]),
                    "validity_rate": float(metric["validity_rate"]),
                    "success_rate": float(metric["success_rate"]) if np.isfinite(metric["success_rate"]) else math.nan,
                    "quality": float(metric["quality"]),
                    "qpu_access_time": float(qpu_access_time) if qpu_access_time is not None else math.nan,
                    "qpu_sampling_time": float(qpu_sampling_time) if qpu_sampling_time is not None else math.nan,
                    "sample_elapsed_sec": float(sample_result.get("elapsed_sec", math.nan)),
                    "mRMSD": float(eval_result.get("mRMSD")) if eval_result.get("mRMSD") is not None else math.nan,
                    "avg_RMSD": float(metric.get("avg_RMSD_report", math.nan)),
                    "physical_qubits": float(sample_result.get("physical_qubits"))
                    if sample_result.get("physical_qubits") is not None
                    else math.nan,
                    "max_chain_length": float(sample_result.get("max_chain_length"))
                    if sample_result.get("max_chain_length") is not None
                    else math.nan,
                    "raw_case_id": case_name,
                }
            )

        return out


def objective_from_metrics(obj_name: str, by_ligand: dict[str, dict[str, Any]]) -> float:
    q3 = float(by_ligand["3nq9"]["quality"])
    q4 = float(by_ligand["4jsz"]["quality"])
    v3 = float(by_ligand["3nq9"]["validity_rate"])
    v4 = float(by_ligand["4jsz"]["validity_rate"])

    if obj_name == "quality_min":
        return float(min(q3, q4))
    if obj_name == "quality_mean":
        return float(0.5 * (q3 + q4))
    if obj_name == "quality_3nq9":
        return q3
    if obj_name == "quality_4jsz":
        return q4
    if obj_name == "robust_score_constrained":
        tau = 0.25
        lam = 0.25
        return float(min(q3, q4) - lam * max(0.0, tau - min(v3, v4)))
    raise ValueError(f"Unknown objective: {obj_name}")


def build_pair_record(
    step: int,
    idx: int,
    kdist: float,
    kmono: float,
    objective_value: float,
    best_so_far: float,
    by_ligand: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    r3 = by_ligand["3nq9"]
    r4 = by_ligand["4jsz"]
    return {
        "step": step,
        "candidate_idx": idx,
        "kdist": kdist,
        "kmono": kmono,
        "objective_value": objective_value,
        "best_so_far": best_so_far,
        "q_3nq9": r3["quality"],
        "q_4jsz": r4["quality"],
        "v_3nq9": r3["validity_rate"],
        "v_4jsz": r4["validity_rate"],
        "success_3nq9": r3["success_rate"],
        "success_4jsz": r4["success_rate"],
        "n_valid_3nq9": r3["n_valid"],
        "n_valid_4jsz": r4["n_valid"],
        "n_success_valid_3nq9": r3["n_success_valid"],
        "n_success_valid_4jsz": r4["n_success_valid"],
        "qpu_access_time_3nq9": r3["qpu_access_time"],
        "qpu_access_time_4jsz": r4["qpu_access_time"],
        "qpu_sampling_time_3nq9": r3["qpu_sampling_time"],
        "qpu_sampling_time_4jsz": r4["qpu_sampling_time"],
        "sample_elapsed_sec_3nq9": r3["sample_elapsed_sec"],
        "sample_elapsed_sec_4jsz": r4["sample_elapsed_sec"],
    }


def matern52_kernel(a: np.ndarray, b: np.ndarray, length_scale: float) -> np.ndarray:
    d = cdist(a, b, metric="euclidean")
    z = np.sqrt(5.0) * d / length_scale
    return (1.0 + z + (z * z) / 3.0) * np.exp(-z)


@dataclass
class GPModel:
    x_train: np.ndarray
    y_mean: float
    y_std: float
    length_scale: float
    chol: np.ndarray
    alpha: np.ndarray


def fit_gp_m52(x_train: np.ndarray, y_train: np.ndarray) -> GPModel:
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-12:
        y_std = 1.0
    y = (y_train - y_mean) / y_std

    cand_l = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.8, 1.2], dtype=float)
    best = None
    best_lml = -np.inf
    n = x_train.shape[0]
    eye = np.eye(n)

    for l in cand_l:
        k = matern52_kernel(x_train, x_train, l)
        jitter = 1e-8
        chol = None
        for _ in range(6):
            try:
                chol = np.linalg.cholesky(k + (1e-6 + jitter) * eye)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if chol is None:
            continue
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        lml = -0.5 * float(y @ alpha) - float(np.sum(np.log(np.diag(chol)))) - 0.5 * n * np.log(2.0 * np.pi)
        if lml > best_lml:
            best_lml = lml
            best = (l, chol, alpha)

    if best is None:
        raise RuntimeError("Failed to fit GP model")

    l, chol, alpha = best
    return GPModel(
        x_train=x_train,
        y_mean=y_mean,
        y_std=y_std,
        length_scale=float(l),
        chol=chol,
        alpha=alpha,
    )


def gp_predict(model: GPModel, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_star = matern52_kernel(model.x_train, x_test, model.length_scale)
    mu_n = k_star.T @ model.alpha
    v = np.linalg.solve(model.chol, k_star)
    var_n = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-12)

    mu = model.y_mean + model.y_std * mu_n
    std = model.y_std * np.sqrt(var_n)
    return mu, std


def expected_improvement(mu: np.ndarray, std: np.ndarray, y_best: float, xi: float) -> np.ndarray:
    imp = mu - y_best - xi
    z = np.zeros_like(imp)
    mask = std > 1e-12
    z[mask] = imp[mask] / std[mask]
    ei = np.zeros_like(imp)
    ei[mask] = imp[mask] * norm.cdf(z[mask]) + std[mask] * norm.pdf(z[mask])
    return ei


def select_next_bo(
    x: np.ndarray,
    observed_idx: list[int],
    observed_values: list[float],
    unobserved_idx: list[int],
    xi: float,
    rng: np.random.Generator,
) -> int:
    train_idx = np.array(observed_idx, dtype=int)
    y_train = np.array(observed_values, dtype=float)
    model = fit_gp_m52(x[train_idx], y_train)

    cand = np.array(sorted(unobserved_idx), dtype=int)
    mu, std = gp_predict(model, x[cand])
    y_best = float(np.max(y_train))
    ei = expected_improvement(mu, std, y_best, xi=xi)

    if np.all(ei <= 1e-16):
        return int(rng.choice(cand))
    mx = float(np.max(ei))
    top = cand[np.where(np.abs(ei - mx) <= 1e-14)[0]]
    return int(rng.choice(top))


CASE_NAME_RE = re.compile(r"^(3nq9|4jsz)_Kdist([0-9.]+)_Kmono([0-9.]+)$")
STEP_TAG_RE = re.compile(r"_s(\d{3})$")


def parse_existing_case_step(report_csv: Path) -> tuple[int, str, float, float, str] | None:
    # .../work/eval/{case_name}/{query_tag}/{ligand}_ligand_report.csv
    try:
        query_tag = report_csv.parent.name
        case_name = report_csv.parent.parent.name
    except Exception:
        return None

    m_step = STEP_TAG_RE.search(query_tag)
    m_case = CASE_NAME_RE.match(case_name)
    if m_step is None or m_case is None:
        return None
    step = int(m_step.group(1))
    ligand = m_case.group(1)
    kdist = float(m_case.group(2))
    kmono = float(m_case.group(3))
    return step, ligand, kdist, kmono, query_tag


def load_existing_qpu_rows(
    out_dir: Path,
    ligands: list[str],
    num_reads: int,
) -> dict[int, dict[str, dict[str, Any]]]:
    eval_root = out_dir / "work" / "eval"
    step_rows: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    if not eval_root.exists():
        return {}

    for report_csv in eval_root.glob("*/*/*_ligand_report.csv"):
        parsed = parse_existing_case_step(report_csv)
        if parsed is None:
            continue
        step, ligand, kdist, kmono, query_tag = parsed
        if ligand not in ligands:
            continue

        metric = compute_metrics_from_report(report_csv, n_reads=num_reads)

        case_name = report_csv.parent.parent.name
        sample_meta_path = (
            out_dir / "work" / "samples" / case_name / query_tag / f"{ligand}_ligand_sampler_meta.json"
        )
        qpu_access_time = math.nan
        qpu_sampling_time = math.nan
        physical_qubits = math.nan
        max_chain_length = math.nan
        sample_elapsed_sec = math.nan
        if sample_meta_path.exists():
            try:
                sm = json.loads(sample_meta_path.read_text(encoding="utf-8"))
                timing = sm.get("timing", {}) or {}
                if timing.get("qpu_access_time") is not None:
                    qpu_access_time = float(timing.get("qpu_access_time"))
                if timing.get("qpu_sampling_time") is not None:
                    qpu_sampling_time = float(timing.get("qpu_sampling_time"))
                if sm.get("physical_qubits") is not None:
                    physical_qubits = float(sm.get("physical_qubits"))
                if sm.get("max_chain_length") is not None:
                    max_chain_length = float(sm.get("max_chain_length"))
                if sm.get("elapsed_sec") is not None:
                    sample_elapsed_sec = float(sm.get("elapsed_sec"))
            except Exception:
                pass

        mRMSD = math.nan
        result_json = report_csv.parent / "result.json"
        if result_json.exists():
            try:
                ej = json.loads(result_json.read_text(encoding="utf-8"))
                if ej.get("mRMSD") is not None:
                    mRMSD = float(ej.get("mRMSD"))
            except Exception:
                pass

        row = {
            "mode": "qpu",
            "query_tag": query_tag,
            "ligand": ligand,
            "kdist": float(kdist),
            "kmono": float(kmono),
            "n_reads": int(metric["n_reads"]),
            "n_valid": int(metric["n_valid"]),
            "n_success_valid": int(metric["n_success_valid"]),
            "validity_rate": float(metric["validity_rate"]),
            "success_rate": float(metric["success_rate"]) if np.isfinite(metric["success_rate"]) else math.nan,
            "quality": float(metric["quality"]),
            "qpu_access_time": qpu_access_time,
            "qpu_sampling_time": qpu_sampling_time,
            "sample_elapsed_sec": sample_elapsed_sec,
            "mRMSD": mRMSD,
            "avg_RMSD": float(metric.get("avg_RMSD_report", math.nan)),
            "physical_qubits": physical_qubits,
            "max_chain_length": max_chain_length,
            "raw_case_id": case_name,
            "_mtime": float(report_csv.stat().st_mtime),
        }
        prev = step_rows[step].get(ligand)
        # Keep the newest artifact when duplicate files exist for the same (step, ligand).
        if prev is None or float(row["_mtime"]) >= float(prev.get("_mtime", -np.inf)):
            step_rows[step][ligand] = row
    return dict(step_rows)


def reconstruct_resume_state(
    out_dir: Path,
    ligands: list[str],
    num_reads: int,
    objective_name: str,
    candidate_to_idx: dict[tuple[float, float], int],
) -> dict[str, Any]:
    step_rows = load_existing_qpu_rows(out_dir=out_dir, ligands=ligands, num_reads=num_reads)
    if not step_rows:
        return {
            "resumed_steps": 0,
            "obs_records": [],
            "trace_records": [],
            "observed_idx": [],
            "observed_values": [],
            "best_so_far": -np.inf,
            "ordered_idx": [],
            "ordered_values": [],
        }

    obs_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    observed_idx: list[int] = []
    observed_values: list[float] = []
    ordered_idx: list[int] = []
    ordered_values: list[float] = []
    best_so_far = -np.inf

    resumed_steps = 0
    used_idx: set[int] = set()
    max_step = max(step_rows) if step_rows else 0
    for step in range(1, max_step + 1):
        rows = step_rows.get(step)
        if rows is None:
            break
        if not all(l in rows for l in ligands):
            break

        pair_set = {pair_key(rows[l]["kdist"], rows[l]["kmono"]) for l in ligands}
        # Stop at the first inconsistent step to keep a valid contiguous prefix.
        if len(pair_set) != 1:
            break
        (kd_key, km_key) = next(iter(pair_set))
        idx = candidate_to_idx.get((kd_key, km_key))
        if idx is None:
            break
        # BO should not query the same discrete candidate twice.
        if idx in used_idx:
            break
        used_idx.add(idx)
        kdist = float(kd_key)
        kmono = float(km_key)

        by_ligand = {lig: rows[lig] for lig in ligands}
        y = float(objective_from_metrics(objective_name, by_ligand))
        observed_idx.append(int(idx))
        observed_values.append(y)
        ordered_idx.append(int(idx))
        ordered_values.append(y)
        best_so_far = float(max(best_so_far, y))

        for lig in ligands:
            row_out = dict(rows[lig])
            row_out["step"] = step
            row_out["candidate_idx"] = int(idx)
            row_out["objective"] = objective_name
            row_out["objective_value"] = y
            row_out["best_so_far"] = best_so_far
            obs_records.append(row_out)

        trace_records.append(
            build_pair_record(
                step=step,
                idx=int(idx),
                kdist=kdist,
                kmono=kmono,
                objective_value=y,
                best_so_far=best_so_far,
                by_ligand=by_ligand,
            )
        )
        resumed_steps = step

    return {
        "resumed_steps": resumed_steps,
        "obs_records": obs_records,
        "trace_records": trace_records,
        "observed_idx": observed_idx,
        "observed_values": observed_values,
        "best_so_far": best_so_far,
        "ordered_idx": ordered_idx,
        "ordered_values": ordered_values,
    }


def advance_rng_for_resumed_prefix(
    rng: np.random.Generator,
    x: np.ndarray,
    n_init: int,
    xi: float,
    completed_idx: list[int],
    completed_obj: list[float],
) -> None:
    # Replay BO random calls so resumed runs keep the RNG progression close to the original run.
    if not completed_idx:
        return
    n_points = x.shape[0]
    _ = list(rng.choice(n_points, size=n_init, replace=False))
    observed_idx_tmp: list[int] = []
    observed_values_tmp: list[float] = []
    remaining_tmp = set(range(n_points))
    for step in range(1, len(completed_idx) + 1):
        if step > n_init:
            _ = select_next_bo(
                x=x,
                observed_idx=observed_idx_tmp,
                observed_values=observed_values_tmp,
                unobserved_idx=sorted(remaining_tmp),
                xi=xi,
                rng=rng,
            )
        idx_actual = int(completed_idx[step - 1])
        y_actual = float(completed_obj[step - 1])
        observed_idx_tmp.append(idx_actual)
        observed_values_tmp.append(y_actual)
        if idx_actual in remaining_tmp:
            remaining_tmp.remove(idx_actual)


def main() -> None:
    args = parse_args()
    ligands = [x.strip().lower() for x in args.pdb_ids.split(",") if x.strip()]
    if set(ligands) != {"3nq9", "4jsz"}:
        raise ValueError("This online BO script currently requires exactly two ligands: 3nq9,4jsz")

    kdist_values = parse_float_list(args.kdist_list) or default_kdist_values()
    kmono_values = parse_float_list(args.kmono_list) or default_kmono_values()

    candidates: list[tuple[float, float]] = []
    for kd in kdist_values:
        for km in kmono_values:
            candidates.append((float(kd), float(km)))
    candidates = sorted(set(candidates))
    n_points = len(candidates)
    candidate_to_idx = {pair_key(kd, km): i for i, (kd, km) in enumerate(candidates)}

    if args.budget > n_points:
        raise ValueError(f"budget={args.budget} exceeds number of candidates={n_points}")
    if args.n_init >= args.budget:
        raise ValueError("n_init must be < budget")

    out_dir = args.out_dir
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "report").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag or datetime.utcnow().strftime("online_bo_%Y%m%d_%H%M%S")

    if args.mode == "oracle":
        evaluator = OracleEvaluator(args.oracle_csv, ligands=ligands, num_reads=args.num_reads)
    else:
        evaluator = QPUEvaluator(
            repo_root=args.repo_root,
            out_dir=out_dir,
            ligands=ligands,
            num_reads=args.num_reads,
            sampler_config=args.sampler_config,
            embed_reuse=args.embed_reuse,
        )

    x_raw = np.array(candidates, dtype=float)
    x_min = x_raw.min(axis=0)
    x_max = x_raw.max(axis=0)
    x = (x_raw - x_min) / (x_max - x_min)

    observed_idx: list[int] = []
    observed_values: list[float] = []
    remaining = set(range(n_points))

    obs_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    best_so_far = -np.inf
    start_step = 1

    if args.resume:
        resume = reconstruct_resume_state(
            out_dir=out_dir,
            ligands=ligands,
            num_reads=args.num_reads,
            objective_name=args.objective,
            candidate_to_idx=candidate_to_idx,
        )
        resumed_steps = int(resume["resumed_steps"])
        if resumed_steps > 0:
            obs_records = list(resume["obs_records"])
            trace_records = list(resume["trace_records"])
            observed_idx = list(resume["observed_idx"])
            observed_values = list(resume["observed_values"])
            best_so_far = float(resume["best_so_far"])
            remaining = set(range(n_points)) - set(observed_idx)
            start_step = resumed_steps + 1
            print(f"[resume] loaded completed steps: {resumed_steps}, continuing from step {start_step}")
        else:
            print("[resume] no completed step found; starting from step 1")

    rng = np.random.default_rng(args.seed)
    initial_idx = list(rng.choice(n_points, size=args.n_init, replace=False))
    if args.resume and start_step > 1:
        advance_rng_for_resumed_prefix(
            rng=rng,
            x=x,
            n_init=args.n_init,
            xi=args.xi,
            completed_idx=list(observed_idx),
            completed_obj=list(observed_values),
        )

    for step in range(start_step, args.budget + 1):
        if step <= args.n_init:
            idx = int(initial_idx[step - 1])
        else:
            if args.policy == "random":
                idx = int(rng.choice(sorted(remaining)))
            else:
                idx = select_next_bo(
                    x=x,
                    observed_idx=observed_idx,
                    observed_values=observed_values,
                    unobserved_idx=sorted(remaining),
                    xi=args.xi,
                    rng=rng,
                )
        # Resume safety: avoid re-querying already observed candidates.
        if idx not in remaining:
            if not remaining:
                raise RuntimeError("No remaining candidates to evaluate")
            if step <= args.n_init:
                alt = next((j for j in initial_idx if j in remaining), None)
                if alt is None:
                    alt = int(rng.choice(sorted(remaining)))
                idx = int(alt)
            else:
                if args.policy == "random":
                    idx = int(rng.choice(sorted(remaining)))
                else:
                    idx = select_next_bo(
                        x=x,
                        observed_idx=observed_idx,
                        observed_values=observed_values,
                        unobserved_idx=sorted(remaining),
                        xi=args.xi,
                        rng=rng,
                    )

        kdist, kmono = candidates[idx]
        query_tag = f"{run_tag}_s{step:03d}"

        last_err = None
        result_rows: list[dict[str, Any]] | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                result_rows = evaluator.evaluate(kdist=kdist, kmono=kmono, query_tag=query_tag)
                last_err = None
                break
            except Exception as exc:  # pragma: no cover - runtime path
                last_err = exc
                if attempt == args.max_retries:
                    break
                continue
        if result_rows is None:
            raise RuntimeError(f"Evaluation failed at step {step}: {type(last_err).__name__}: {last_err}")

        by_ligand = {r["ligand"]: r for r in result_rows}
        y = objective_from_metrics(args.objective, by_ligand)

        observed_idx.append(idx)
        observed_values.append(float(y))
        remaining.discard(idx)
        best_so_far = float(max(best_so_far, y))

        for row in result_rows:
            row_out = dict(row)
            row_out["step"] = step
            row_out["candidate_idx"] = idx
            row_out["objective"] = args.objective
            row_out["objective_value"] = float(y)
            row_out["best_so_far"] = best_so_far
            obs_records.append(row_out)

        trace_records.append(
            build_pair_record(
                step=step,
                idx=idx,
                kdist=kdist,
                kmono=kmono,
                objective_value=float(y),
                best_so_far=best_so_far,
                by_ligand=by_ligand,
            )
        )

        print(
            f"[{step:02d}/{args.budget}] k=({kdist:.3f},{kmono:.3f}) "
            f"obj={y:.6f} best={best_so_far:.6f}",
            flush=True,
        )

    obs_df = pd.DataFrame(obs_records)
    trace_df = pd.DataFrame(trace_records)

    obs_path = out_dir / "results" / "online_bo_observations.csv"
    trace_path = out_dir / "results" / "online_bo_trace.csv"
    obs_df.to_csv(obs_path, index=False)
    trace_df.to_csv(trace_path, index=False)

    fig_path = out_dir / "figures" / "online_bo_best_so_far.png"
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(trace_df["step"], trace_df["best_so_far"], color="#1f77b4", linewidth=2.2)
    plt.scatter(trace_df["step"], trace_df["objective_value"], color="#ff7f0e", s=24, alpha=0.85, label="Observed")
    plt.xlabel("BO query step")
    plt.ylabel(args.objective)
    plt.title(f"Online BO ({args.mode}) - best-so-far")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=args.plot_dpi)
    plt.close()

    best_row = trace_df.sort_values("objective_value", ascending=False).iloc[0]

    report_path = out_dir / "report" / "online_bo_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Online BO Report\n\n")
        f.write(f"- mode: `{args.mode}`\n")
        f.write(f"- objective: `{args.objective}`\n")
        f.write(f"- run_tag: `{run_tag}`\n")
        f.write(f"- budget: `{args.budget}` (init `{args.n_init}`)\n")
        f.write(f"- policy: `{args.policy}`\n")
        f.write(f"- seed: `{args.seed}`\n")
        f.write(f"- xi: `{args.xi}`\n")
        f.write(f"- num_reads: `{args.num_reads}`\n")
        if args.mode == "qpu":
            f.write(f"- sampler_config: `{args.sampler_config}`\n")
            f.write(f"- embed_reuse: `{args.embed_reuse}`\n")
        else:
            f.write(f"- oracle_csv: `{args.oracle_csv}`\n")

        f.write("\n## Best observed query\n\n")
        f.write(
            f"- step: `{int(best_row['step'])}`\n"
            f"- (kdist, kmono): `({best_row['kdist']:.3f}, {best_row['kmono']:.3f})`\n"
            f"- objective: `{best_row['objective_value']:.6f}`\n"
            f"- q_3nq9 / q_4jsz: `{best_row['q_3nq9']:.6f}` / `{best_row['q_4jsz']:.6f}`\n"
            f"- v_3nq9 / v_4jsz: `{best_row['v_3nq9']:.6f}` / `{best_row['v_4jsz']:.6f}`\n"
        )

        f.write("\n## Notes\n\n")
        f.write(
            "- This is an online sequential BO run: each point is evaluated before selecting the next point.\n"
            "- `quality_min` is the manuscript-aligned quality-centric robust objective.\n"
        )

    cfg = {
        "run_tag": run_tag,
        "mode": args.mode,
        "repo_root": str(args.repo_root),
        "out_dir": str(out_dir),
        "objective": args.objective,
        "budget": args.budget,
        "n_init": args.n_init,
        "policy": args.policy,
        "seed": args.seed,
        "xi": args.xi,
        "num_reads": args.num_reads,
        "embed_reuse": args.embed_reuse,
        "resume": bool(args.resume),
        "sampler_config": str(args.sampler_config),
        "oracle_csv": str(args.oracle_csv),
        "ligands": ligands,
        "kdist_values": kdist_values,
        "kmono_values": kmono_values,
        "n_candidates": n_points,
        "outputs": {
            "observations_csv": str(obs_path),
            "trace_csv": str(trace_path),
            "report_md": str(report_path),
            "figure_best_so_far": str(fig_path),
        },
    }
    cfg_path = out_dir / "logs" / "online_bo_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Wrote:")
    print("-", obs_path)
    print("-", trace_path)
    print("-", fig_path)
    print("-", report_path)
    print("-", cfg_path)


if __name__ == "__main__":
    main()
