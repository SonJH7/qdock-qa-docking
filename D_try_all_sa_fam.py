#!/usr/bin/env python3
import os
import sys
import shutil
import json
import re
import statistics
import subprocess
from datetime import datetime
import time
import numpy as np


def load_sampler_from_env():
    cfg_path = os.environ.get("QDOCK_SAMPLER_CONFIG")
    if not cfg_path:
        return None, None
    from QDock.samplers.D_factory import SamplerFactory, load_config
    cfg = load_config(cfg_path)
    sampler = SamplerFactory.from_config(cfg)
    sampler_kwargs = cfg.get("sampler_kwargs", {}) or {}
    if "num_reads" in cfg and "num_reads" not in sampler_kwargs:
        sampler_kwargs["num_reads"] = cfg["num_reads"]
    return sampler, sampler_kwargs

SAMPLER, SAMPLER_KWARGS = load_sampler_from_env()


def rmsd(pose, ref_coords):
    diff = pose - ref_coords
    return float(np.sqrt((diff * diff).sum() / len(ref_coords)))


def load_pdb_ids(base_dir, fallback_ids):
    ids_path = os.path.join(base_dir, "t2_ids.txt")
    if not os.path.exists(ids_path):
        return fallback_ids, None
    ids = []
    with open(ids_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0].lower()
            if len(token) == 4:
                ids.append(token)
    # Dedup while preserving order
    seen = set()
    ordered = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    return ordered, ids_path


def count_gasteiger_warnings(text):
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if "Gasteiger parameters" in line)


def classify_error(exc):
    msg = str(exc)
    lower = msg.lower()
    if isinstance(exc, FileNotFoundError) and "prepare_receptor" in msg:
        return "prepare_receptor_error", msg
    if "prepare_receptor" in lower:
        return "prepare_receptor_error", msg
    if "autosite" in lower or isinstance(exc, IndexError):
        return "autosite_error", msg
    if "robin_hood::map overflow" in msg or "overflow" in lower:
        return "qubo_overflow", msg
    return "unknown_error", msg


def get_tool_version(name, patterns):
    path = shutil.which(name)
    if not path:
        return None, None
    for args in (["--version"], ["-h"]):
        try:
            res = subprocess.run([name] + args, capture_output=True, text=True)
        except Exception:
            continue
        text = (res.stdout or "") + "\n" + (res.stderr or "")
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return path, match.group(1)
    return path, None


def summarize_times(values):
    if not values:
        return {"avg": None, "median": None, "max": None, "total": None}
    return {
        "avg": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
        "total": round(sum(values), 3),
    }


def run_case(pdb_id, base_dir, work_root, params):
    start = time.perf_counter()
    start_cpu = time.process_time()
    case_dir = os.path.join(work_root, pdb_id)
    os.makedirs(case_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(case_dir)

    receptor_path = os.path.join(base_dir, "data", f"{pdb_id}_protein.pdb")
    ligand_name = f"{pdb_id}_ligand.mol2"
    ligand_path = os.path.join(base_dir, "data", ligand_name)

    result = {
        "pdb_id": pdb_id,
        "params": {
            "edge_cutoff": params["edge_cutoff"],
            "K_dist": params["K_dist"],
            "K_mono": params["K_mono"],
            "n_pos": params["n_pos"],
            "grid_length": params["grid_length"],
            "max_qubits": params["max_qubits"],
            "max_terms": params["max_terms"],
            "seed": params["seed"],
        },
        "paths": {
            "result_dir": case_dir,
            "qubo_dir": os.path.join(case_dir, "QUBOs"),
            "matches_dir": os.path.join(case_dir, "Matches"),
            "poses_dir": os.path.join(case_dir, "Poses"),
            "poses_file": os.path.join(case_dir, "Poses", f"{pdb_id}_ligand_poses.pdb"),
            "pocs_dir": os.path.join(case_dir, "pocs"),
            "autosite_log": os.path.join(case_dir, "pocs_AutoSiteSummary.log"),
            "receptor_pdbqt": os.path.join(case_dir, "receptor.pdbqt"),
            "ligand_pdbqt": os.path.join(case_dir, "Ligands", f"{pdb_id}_ligand.pdbqt"),
        },
    }
    try:
        if not os.path.exists(receptor_path):
            result.update(
                {
                    "status": "missing_receptor",
                    "error_type": "missing_receptor",
                    "error_msg": "receptor file not found",
                }
            )
            return result
        if not os.path.exists(ligand_path):
            result.update(
                {
                    "status": "missing_ligand",
                    "error_type": "missing_ligand",
                    "error_msg": "ligand file not found",
                }
            )
            return result

        from QDock.FeatureAtomMatching.D_qdock import FAMDock

        fam = FAMDock()
        fam.make_receptor(receptor_path)
        gasteiger_count = count_gasteiger_warnings(
            getattr(fam.receptor, "prepare_receptor_stdout", "")
        ) + count_gasteiger_warnings(getattr(fam.receptor, "prepare_receptor_stderr", ""))

        shutil.copy(ligand_path, ligand_name)
        fam.make_ligand([ligand_name])
        fam.make_box_ligand(ligand_name, grid_length=params["grid_length"])

        feature_atoms = len(fam.feature_atoms)
        ligand_atoms = fam.ligands[0].n
        qubits = feature_atoms * ligand_atoms
        estimated_terms = int(qubits * (qubits - 1) / 2)

        if qubits > params["max_qubits"]:
            result.update(
                {
                    "status": "skipped_too_large",
                    "error_type": "skipped_too_large",
                    "ligand_atoms": ligand_atoms,
                    "feature_atoms": feature_atoms,
                    "qubits": qubits,
                    "estimated_terms": estimated_terms,
                    "gasteiger_warning_count": gasteiger_count,
                }
            )
            return result
        if estimated_terms > params["max_terms"]:
            result.update(
                {
                    "status": "skipped_too_many_terms",
                    "error_type": "skipped_too_many_terms",
                    "ligand_atoms": ligand_atoms,
                    "feature_atoms": feature_atoms,
                    "qubits": qubits,
                    "estimated_terms": estimated_terms,
                    "gasteiger_warning_count": gasteiger_count,
                }
            )
            return result

        poses = fam.indiv_dock(
            ligand=fam.ligands[0],
            edge_cutoff=params["edge_cutoff"],
            K_dist=params["K_dist"],
            K_mono=params["K_mono"],
            n_pos=params["n_pos"],
            save_qubo=True,
            sim_dock=True,
            save_match=True,
            save_pose=True,
            sampler=SAMPLER,
            sampler_kwargs=SAMPLER_KWARGS,
        )

        m_rmsd = None
        n_poses = 0
        if isinstance(poses, np.ndarray) and poses.size > 0:
            if poses.ndim == 2:
                poses = poses[None, ...]
            n_poses = poses.shape[0]
            ref_coords = fam.ligands[0].coords
            rmsds = [rmsd(pose, ref_coords) for pose in poses]
            m_rmsd = float(min(rmsds))

        result.update(
            {
                "status": "ok",
                "ligand_atoms": ligand_atoms,
                "feature_atoms": feature_atoms,
                "qubits": qubits,
                "estimated_terms": estimated_terms,
                "n_poses": n_poses,
                "mRMSD": m_rmsd,
                "gasteiger_warning_count": gasteiger_count,
            }
        )
        return result
    except Exception as exc:
        error_type, error_msg = classify_error(exc)
        result.update(
            {
                "status": "error",
                "error_type": error_type,
                "error_msg": error_msg,
            }
        )
        return result
    finally:
        result["elapsed_sec"] = round(time.perf_counter() - start, 3)
        result["cpu_sec"] = round(time.process_time() - start_cpu, 3)
        os.chdir(prev_cwd)


def build_summary(results, total_cases, success_threshold, env_info, run_config):
    ok_cases = [r for r in results if r.get("status") == "ok" and r.get("mRMSD") is not None]
    success_count = sum(1 for r in ok_cases if r["mRMSD"] <= success_threshold)
    success_rate = (success_count / len(ok_cases)) if ok_cases else None
    success_rate_all = (success_count / total_cases) if total_cases else None

    error_counts = {
        "missing_receptor": 0,
        "missing_ligand": 0,
        "skipped_too_large": 0,
        "skipped_too_many_terms": 0,
        "prepare_receptor_error": 0,
        "autosite_error": 0,
        "qubo_overflow": 0,
        "unknown_error": 0,
    }
    for r in results:
        status = r.get("status")
        if status in (
            "missing_receptor",
            "missing_ligand",
            "skipped_too_large",
            "skipped_too_many_terms",
        ):
            error_counts[status] += 1
            continue
        if status == "error":
            error_type = r.get("error_type", "unknown_error")
            if error_type not in error_counts:
                error_counts["unknown_error"] += 1
            else:
                error_counts[error_type] += 1

    elapsed_values = [r["elapsed_sec"] for r in results if r.get("elapsed_sec") is not None]
    time_stats = summarize_times(elapsed_values)
    cpu_values = [r["cpu_sec"] for r in results if r.get("cpu_sec") is not None]
    cpu_stats = summarize_times(cpu_values)

    gasteiger_counts = [r.get("gasteiger_warning_count", 0) for r in results]
    gasteiger_total = int(sum(gasteiger_counts))
    gasteiger_cases = sum(1 for c in gasteiger_counts if c)

    qubits_values = [r["qubits"] for r in results if r.get("qubits") is not None]
    qubits_avg = round(statistics.mean(qubits_values), 3) if qubits_values else None
    qubits_max = max(qubits_values) if qubits_values else None

    terms_values = [r["estimated_terms"] for r in results if r.get("estimated_terms") is not None]
    terms_avg = round(statistics.mean(terms_values), 3) if terms_values else None

    return {
        "total_cases": total_cases,
        "success_threshold": success_threshold,
        "success_count": success_count,
        "success_total": len(ok_cases),
        "success_rate": success_rate,
        "success_rate_all": success_rate_all,
        "error_counts": error_counts,
        "elapsed_sec_avg": time_stats["avg"],
        "elapsed_sec_median": time_stats["median"],
        "elapsed_sec_max": time_stats["max"],
        "elapsed_sec_total": time_stats["total"],
        "cpu_sec_avg": cpu_stats["avg"],
        "cpu_sec_median": cpu_stats["median"],
        "cpu_sec_max": cpu_stats["max"],
        "cpu_sec_total": cpu_stats["total"],
        "gasteiger_warning_total": gasteiger_total,
        "gasteiger_warning_cases": gasteiger_cases,
        "qubits_avg": qubits_avg,
        "qubits_max": qubits_max,
        "estimated_terms_avg": terms_avg,
        "environment": env_info,
        "run_config": run_config,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    # Optional directory containing autosite, prepare_receptor, and prepare_ligand.
    adfr_bin = os.environ.get("QDOCK_ADFR_BIN")
    if adfr_bin:
        os.environ["PATH"] = adfr_bin + os.pathsep + os.environ.get("PATH", "")

    base_work_root = os.path.join(base_dir, "try_all_sa_workdir_fam")
    os.makedirs(base_work_root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_root = os.path.join(base_work_root, run_id)
    os.makedirs(work_root, exist_ok=True)

    params = {
        "edge_cutoff": 1.870,
        "K_dist": 2.261,
        "K_mono": 11.479,
        "n_pos": 30,
        "grid_length": 1.0,
        "max_qubits": 2500,  # skip very large cases to avoid QUBO overflow
        "max_terms": 100000,
        "seed": 42,
    }

    pdb_ids, ids_source = load_pdb_ids(
        base_dir, ["1y6r", "2vkm", "3dxg", "2iwx"]
    )
    if ids_source:
        print(f"Loaded {len(pdb_ids)} IDs from {ids_source}")
    else:
        print(f"Using default IDs: {pdb_ids}")
    success_threshold = 2.0
    autosite_path, autosite_version = get_tool_version(
        "autosite", [r"AutoSite v([0-9.]+)"]
    )
    prepare_path, adfr_version = get_tool_version(
        "prepare_receptor", [r"AutoDockTools[^0-9]*([0-9.]+)", r"prepare_receptor[^0-9]*([0-9.]+)"]
    )
    env_info = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "adfr_bin": adfr_bin,
        "prepare_receptor_path": prepare_path,
        "prepare_receptor_version": adfr_version,
        "autosite_path": autosite_path,
        "autosite_version": autosite_version,
    }
    run_config = {
        "seed": params["seed"],
        "n_pos": params["n_pos"],
        "grid_length": params["grid_length"],
        "edge_cutoff": params["edge_cutoff"],
        "K_dist": params["K_dist"],
        "K_mono": params["K_mono"],
        "max_qubits": params["max_qubits"],
        "max_terms": params["max_terms"],
        "ids_source": ids_source,
    }
    results = []
    out_path = os.path.join(work_root, "try_all_sa_fam_results.json")
    for pdb_id in pdb_ids:
        print(f"[{pdb_id}] running...")
        result = run_case(pdb_id, base_dir, work_root, params)
        results.append(result)
        if result.get("status") == "ok":
            print(
                f"{result['pdb_id']} | ligand_atoms={result['ligand_atoms']} | "
                f"feature_atoms={result['feature_atoms']} | qubits={result['qubits']} | "
                f"n_poses={result['n_poses']} | mRMSD={result['mRMSD']} | "
                f"elapsed_sec={result.get('elapsed_sec')}"
            )
        elif result.get("status") == "skipped_too_large":
            print(
                f"{result['pdb_id']}: skipped_too_large "
                f"(qubits={result.get('qubits')}, max_qubits={params['max_qubits']}, "
                f"elapsed_sec={result.get('elapsed_sec')})"
            )
        elif result.get("status") == "skipped_too_many_terms":
            print(
                f"{result['pdb_id']}: skipped_too_many_terms "
                f"(estimated_terms={result.get('estimated_terms')}, "
                f"max_terms={params['max_terms']}, "
                f"elapsed_sec={result.get('elapsed_sec')})"
            )
        else:
            print(f"{result['pdb_id']}: {result['status']} (elapsed_sec={result.get('elapsed_sec')})")
        summary = build_summary(
            results, len(pdb_ids), success_threshold, env_info, run_config
        )
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)

    summary = build_summary(
        results, len(pdb_ids), success_threshold, env_info, run_config
    )

    print("\n=== Summary ===")
    for row in results:
        if row.get("status") != "ok":
            print(f"{row['pdb_id']}: {row['status']} (elapsed_sec={row.get('elapsed_sec')})")
            continue
        print(
            f"{row['pdb_id']} | ligand_atoms={row['ligand_atoms']} | "
            f"feature_atoms={row['feature_atoms']} | qubits={row['qubits']} | "
            f"n_poses={row['n_poses']} | mRMSD={row['mRMSD']} | "
            f"elapsed_sec={row.get('elapsed_sec')}"
        )
    print(
        f"\nsuccess_rate={summary['success_rate']} "
        f"(threshold={success_threshold}, "
        f"{summary['success_count']}/{summary['success_total']})"
    )
    print(f"success_rate_all={summary['success_rate_all']} (total={summary['total_cases']})")
    print(f"elapsed_sec_total={summary['elapsed_sec_total']}")
    print(f"cpu_sec_total={summary['cpu_sec_total']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
