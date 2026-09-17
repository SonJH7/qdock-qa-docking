#!/usr/bin/env python3
import os
import sys
import json
import time
import statistics
from datetime import datetime

import numpy as np
import prody


def rmsd(pose, ref_coords):
    diff = pose - ref_coords
    return float(np.sqrt((diff * diff).sum() / len(ref_coords)))


def summarize_times(values):
    if not values:
        return {"avg": None, "median": None, "max": None, "total": None}
    return {
        "avg": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
        "total": round(sum(values), 3),
    }


def pick_latest_run(base_root):
    if not os.path.isdir(base_root):
        return None
    candidates = [
        os.path.join(base_root, d)
        for d in os.listdir(base_root)
        if os.path.isdir(os.path.join(base_root, d)) and not d.startswith(".")
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def get_run_tag():
    tag = os.environ.get("QDOCK_RUN_TAG")
    if tag:
        return tag
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_case_root(root, pdb_id, run_tag=None):
    base = os.path.join(root, pdb_id)
    if run_tag:
        return os.path.join(base, run_tag), run_tag
    if not os.path.isdir(base):
        return base, None
    subdirs = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
    ]
    if not subdirs:
        return base, None
    numeric = [d for d in subdirs if d[:8].isdigit()]
    if numeric:
        chosen = sorted(numeric)[-1]
        return os.path.join(base, chosen), chosen
    chosen = sorted(subdirs)[-1]
    return os.path.join(base, chosen), chosen


def filter_case_dirs(case_dirs):
    ids_env = os.environ.get("QDOCK_PDB_IDS")
    if ids_env:
        wanted = {item.strip().lower() for item in ids_env.split(",") if item.strip()}
        case_dirs = [d for d in case_dirs if os.path.basename(d).lower() in wanted]
    limit_env = os.environ.get("QDOCK_LIMIT")
    if limit_env:
        try:
            limit_val = int(limit_env)
        except Exception:
            limit_val = None
        if limit_val is not None:
            case_dirs = case_dirs[:limit_val]
    return case_dirs

def load_samples(samples_path):
    with open(samples_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data.get("samples") or []
    if isinstance(data, list):
        return data
    return []


def load_embedding_stats(sample_dir, ligand_name):
    if not sample_dir or not ligand_name:
        return None, None, None
    path = os.path.join(sample_dir, "%s_embedding.json" % ligand_name)
    if not os.path.exists(path):
        return None, None, None
    try:
        embedding = json.load(open(path))
    except Exception:
        return None, None, None
    if not isinstance(embedding, dict) or not embedding:
        return None, None, None
    try:
        physical_qubits = sum(len(v) for v in embedding.values())
        max_chain = max(len(v) for v in embedding.values())
    except Exception:
        return path, None, None
    return path, physical_qubits, max_chain


def find_ligand_name(qubo_dir):
    for name in sorted(os.listdir(qubo_dir)):
        if name.endswith(".npy"):
            return name[:-4]
    for name in sorted(os.listdir(qubo_dir)):
        if name.endswith("_samples.json"):
            return name[:-13]
    return None


def load_baseline_results(work_root):
    results_path = os.path.join(work_root, "try_all_sa_fam_results.json")
    if not os.path.exists(results_path):
        return {}, None, None
    try:
        data = json.load(open(results_path))
    except Exception:
        return {}, results_path, None
    run_cfg = data.get("summary", {}).get("run_config", {})
    gasteiger_map = {}
    for entry in data.get("results", []):
        pdb_id = entry.get("pdb_id")
        if pdb_id:
            gasteiger_map[pdb_id] = entry.get("gasteiger_warning_count")
    return gasteiger_map, results_path, run_cfg


def report_docking_analysis(docked_poses, native_path, original_coords, elapsed_sec=None, csv_path=None, txt_path=None):
    if not os.path.exists(native_path):
        print("[WARN] Native PDB not found: %s" % native_path)
        return None

    import pandas as pd

    native = prody.parsePDB(native_path)
    native_coords = native.getCoords()
    results_data = []

    for i, pose_coords in enumerate(docked_poses):
        rmsd_val = prody.calcRMSD(native_coords, pose_coords)
        res = prody.superpose(original_coords, pose_coords)
        mapped_orig = res[0]
        shape_diff = prody.calcRMSD(mapped_orig, pose_coords)
        is_feasible = shape_diff < 0.1
        is_success = rmsd_val <= 2.0
        results_data.append({
            "Pose_ID": i + 1,
            "RMSD(A)": round(rmsd_val, 4),
            "Shape_Diff(A)": round(shape_diff, 8),
            "Feasible": is_feasible,
            "Success(<=2.0A)": is_success,
        })

    df = pd.DataFrame(results_data)
    total = len(df)
    feasible_count = df["Feasible"].sum() if total else 0
    success_count = df["Success(<=2.0A)"].sum() if total else 0

    lines = []
    lines.append("Detailed analysis (total %d samples)" % total)
    lines.append("-" * 60)
    lines.append(df.to_string(index=False))
    lines.append("-" * 60)
    if total:
        lines.append("1. Feasibility Rate : %d/%d (%.1f%%)" % (
            feasible_count, total, (feasible_count / total) * 100))
        lines.append("2. Sampling Success Rate (<=2.0A) : %d/%d (%.1f%%)" % (
            success_count, total, (success_count / total) * 100))
        lines.append("3. Minimum RMSD (mRMSD)           : %.4f A" % (df["RMSD(A)"].min()))
        lines.append("4. Average RMSD                   : %.4f A" % (df["RMSD(A)"].mean()))
        if elapsed_sec is not None:
            lines.append("5. Computation Time               : %.3f sec" % elapsed_sec)
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print("\n" + report_text)
    if csv_path:
        df.to_csv(csv_path, index=False)
    if txt_path:
        with open(txt_path, "w") as f:
            f.write(report_text + "\n")
    return df


def build_feature_atoms(pocs_dir):
    pocs = []
    for name in sorted(os.listdir(pocs_dir)):
        if "_fp_" not in name:
            continue
        pocs.append(prody.parsePDB(os.path.join(pocs_dir, name)))
    if not pocs:
        return None
    feature_atoms = pocs[0]
    for i in range(1, len(pocs)):
        feature_atoms += pocs[i]
    return feature_atoms


def save_case_result(result, output_case_dir):
    os.makedirs(output_case_dir, exist_ok=True)
    result_json = os.path.join(output_case_dir, "result.json")
    result_txt = os.path.join(output_case_dir, "result.txt")
    with open(result_json, "w") as f:
        json.dump(result, f, indent=2)
    with open(result_txt, "w") as f:
        json.dump(result, f, indent=2)


def run_case(source_case_dir, sample_root, output_root, params, gasteiger_map, report=False, run_tag=None, output_tag=None):
    start = time.perf_counter()
    pdb_id = os.path.basename(source_case_dir)
    qubo_dir = os.path.join(source_case_dir, "QUBOs")
    pocs_dir = os.path.join(source_case_dir, "pocs")
    sample_case_root, inferred_tag = resolve_case_root(sample_root, pdb_id, run_tag)
    output_case_tag = inferred_tag or output_tag or get_run_tag()
    output_case_dir = os.path.join(output_root, pdb_id, output_case_tag)
    sample_qubo_dir = os.path.join(sample_case_root, "QUBOs")
    skip_existing = os.environ.get("QDOCK_EVAL_SKIP_EXISTING") == "1"
    if skip_existing and os.path.exists(os.path.join(output_case_dir, "result.json")):
        return {
            "pdb_id": pdb_id,
            "run_tag": output_case_tag,
            "status": "skipped",
            "skip_reason": "existing_result",
            "paths": {
                "source_dir": source_case_dir,
                "output_dir": output_case_dir,
                "sample_root": sample_case_root,
                "sample_qubo_dir": sample_qubo_dir,
            },
            "elapsed_sec": 0.0,
        }
    os.makedirs(output_case_dir, exist_ok=True)

    result = {
        "pdb_id": pdb_id,
        "run_tag": output_case_tag,
        "params": dict(params),
        "paths": {
            "source_dir": source_case_dir,
            "sample_root": sample_case_root,
            "output_dir": output_case_dir,
            "qubo_dir": qubo_dir,
            "sample_qubo_dir": sample_qubo_dir,
            "matches_dir": os.path.join(output_case_dir, "Matches"),
            "poses_dir": os.path.join(output_case_dir, "Poses"),
            "pocs_dir": pocs_dir,
        },
    }

    try:
        if not os.path.isdir(qubo_dir):
            result.update({"status": "error", "error_type": "missing_qubo", "error_msg": "QUBOs dir not found"})
            return result

        ligand_name = None
        active_sample_dir = None
        if os.path.isdir(sample_qubo_dir):
            ligand_name = find_ligand_name(sample_qubo_dir)
            if ligand_name:
                active_sample_dir = sample_qubo_dir
        if not ligand_name:
            ligand_name = find_ligand_name(qubo_dir)
            if ligand_name:
                active_sample_dir = qubo_dir
        if not ligand_name:
            result.update({"status": "error", "error_type": "missing_samples", "error_msg": "samples not found"})
            return result

        samples_path = os.path.join(active_sample_dir, "%s_samples.json" % ligand_name)
        result["paths"]["samples_path"] = samples_path
        embed_path, physical_qubits, max_chain = load_embedding_stats(active_sample_dir, ligand_name)
        if embed_path:
            result["paths"]["embedding_path"] = embed_path
        if physical_qubits is not None:
            result["physical_qubits"] = physical_qubits
        if max_chain is not None:
            result["max_chain_length"] = max_chain
        if not os.path.exists(samples_path):
            result.update({"status": "error", "error_type": "missing_samples", "error_msg": "samples file not found"})
            return result

        ligand_pdb = os.path.join(source_case_dir, "Ligands", "%s.pdb" % ligand_name)
        ligand_pdbqt = os.path.join(source_case_dir, "Ligands", "%s.pdbqt" % ligand_name)
        result["paths"]["ligand_pdb"] = ligand_pdb
        result["paths"]["ligand_pdbqt"] = ligand_pdbqt

        if not os.path.exists(ligand_pdb) and not os.path.exists(ligand_pdbqt):
            result.update({"status": "error", "error_type": "missing_ligand", "error_msg": "ligand file not found"})
            return result

        if not os.path.isdir(pocs_dir):
            result.update({"status": "error", "error_type": "missing_pocs", "error_msg": "pocs dir not found"})
            return result

        feature_atoms = build_feature_atoms(pocs_dir)
        if feature_atoms is None:
            result.update({"status": "error", "error_type": "missing_pocs", "error_msg": "pocs parse failed"})
            return result
        feature_coords = feature_atoms.getCoords()

        ligand_struct = prody.parsePDB(ligand_pdb) if os.path.exists(ligand_pdb) else None
        if ligand_struct is None:
            ligand_struct = prody.parsePDB(ligand_pdbqt)
        ligand_coords = ligand_struct.getCoords()

        samples = load_samples(samples_path)
        if not samples:
            result.update({"status": "error", "error_type": "missing_samples", "error_msg": "no samples loaded"})
            return result

        news = []
        match = []
        for sample in samples:
            ls = []
            gs = []
            this_match = []
            for key, val in sample.items():
                try:
                    if int(val) != 1:
                        continue
                except Exception:
                    continue
                parts = key.split("_", 2)
                if len(parts) < 2:
                    continue
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                except Exception:
                    continue
                if i < 0 or j < 0:
                    continue
                if i >= len(ligand_coords) or j >= len(feature_coords):
                    continue
                ls.append(ligand_coords[i])
                gs.append(feature_coords[j])
                this_match.append(key)
            if not ls:
                continue
            try:
                ls = np.vstack(ls)
                gs = np.vstack(gs)
            except Exception:
                continue
            try:
                trans = prody.superpose(ls, gs)[1]
            except Exception:
                continue
            news.append(prody.applyTransformation(trans, ligand_coords))
            match.append(this_match)

        if match:
            os.makedirs(result["paths"]["matches_dir"], exist_ok=True)
            np.save(os.path.join(result["paths"]["matches_dir"], "%s_match.npy" % ligand_name), match)

        if news:
            os.makedirs(result["paths"]["poses_dir"], exist_ok=True)
            tmp = ligand_struct.copy()
            tmp.setCoords(news[0])
            if len(news) > 1:
                tmp.addCoordset(np.array(news[1:]))
            poses_path = os.path.join(result["paths"]["poses_dir"], "%s_poses.pdb" % ligand_name)
            prody.writePDB(poses_path, tmp)
            result["paths"]["poses_file"] = poses_path

        n_poses = len(news)
        m_rmsd = None
        if news:
            rmsds = [rmsd(pose, ligand_coords) for pose in news]
            m_rmsd = float(min(rmsds))

        ligand_atoms = int(ligand_coords.shape[0])
        feature_atoms_count = int(feature_coords.shape[0])
        qubits = ligand_atoms * feature_atoms_count
        estimated_terms = int(qubits * (qubits - 1) / 2)

        gasteiger_count = gasteiger_map.get(pdb_id)
        result.update({
            "status": "ok",
            "ligand_atoms": ligand_atoms,
            "feature_atoms": feature_atoms_count,
            "qubits": qubits,
            "estimated_terms": estimated_terms,
            "n_poses": n_poses,
            "mRMSD": m_rmsd,
            "gasteiger_warning_count": gasteiger_count,
        })

        if report and news:
            native_path = ligand_pdb if os.path.exists(ligand_pdb) else ligand_pdbqt
            elapsed_for_report = round(time.perf_counter() - start, 3)
            os.makedirs(output_case_dir, exist_ok=True)
            report_csv = os.path.join(output_case_dir, "%s_report.csv" % ligand_name)
            report_txt = os.path.join(output_case_dir, "%s_report.txt" % ligand_name)
            result["paths"]["report_csv"] = report_csv
            result["paths"]["report_txt"] = report_txt
            report_docking_analysis(
                np.array(news),
                native_path,
                ligand_coords,
                elapsed_sec=elapsed_for_report,
                csv_path=report_csv,
                txt_path=report_txt,
            )

        return result
    except Exception as exc:
        result.update({"status": "error", "error_type": "unknown_error", "error_msg": str(exc)})
        return result
    finally:
        result["elapsed_sec"] = round(time.perf_counter() - start, 3)
        try:
            save_case_result(result, output_case_dir)
        except Exception:
            pass


def build_summary(results, total_cases, success_threshold, env_info, run_config):
    ok_cases = [r for r in results if r.get("status") == "ok" and r.get("mRMSD") is not None]
    success_count = sum(1 for r in ok_cases if r["mRMSD"] <= success_threshold)
    success_rate = (success_count / len(ok_cases)) if ok_cases else None
    success_rate_all = (success_count / total_cases) if total_cases else None

    error_counts = {
        "missing_qubo": 0,
        "missing_samples": 0,
        "missing_ligand": 0,
        "missing_pocs": 0,
        "unknown_error": 0,
        "skipped": 0,
    }
    for r in results:
        if r.get("status") == "skipped":
            error_counts["skipped"] += 1
            continue
        if r.get("status") != "error":
            continue
        error_type = r.get("error_type", "unknown_error")
        if error_type not in error_counts:
            error_counts["unknown_error"] += 1
        else:
            error_counts[error_type] += 1

    elapsed_values = [r["elapsed_sec"] for r in results if r.get("elapsed_sec") is not None]
    time_stats = summarize_times(elapsed_values)

    gasteiger_values = [r.get("gasteiger_warning_count") for r in results]
    gasteiger_values = [v for v in gasteiger_values if isinstance(v, int)]
    gasteiger_total = sum(gasteiger_values) if gasteiger_values else None
    gasteiger_cases = sum(1 for v in gasteiger_values if v > 0) if gasteiger_values else None

    qubits_values = [r["qubits"] for r in ok_cases if r.get("qubits") is not None]
    qubits_avg = round(statistics.mean(qubits_values), 3) if qubits_values else None
    qubits_max = max(qubits_values) if qubits_values else None

    terms_values = [r["estimated_terms"] for r in ok_cases if r.get("estimated_terms") is not None]
    terms_avg = round(statistics.mean(terms_values), 3) if terms_values else None

    physical_values = [r["physical_qubits"] for r in ok_cases if r.get("physical_qubits") is not None]
    physical_avg = round(statistics.mean(physical_values), 3) if physical_values else None
    physical_max = max(physical_values) if physical_values else None
    chain_values = [r["max_chain_length"] for r in ok_cases if r.get("max_chain_length") is not None]
    chain_max = max(chain_values) if chain_values else None

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
        "gasteiger_warning_total": gasteiger_total,
        "gasteiger_warning_cases": gasteiger_cases,
        "qubits_avg": qubits_avg,
        "qubits_max": qubits_max,
        "estimated_terms_avg": terms_avg,
        "physical_qubits_avg": physical_avg,
        "physical_qubits_max": physical_max,
        "max_chain_length_max": chain_max,
        "environment": env_info,
        "run_config": run_config,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    base_work_root = os.path.join(base_dir, "try_all_sa_workdir_fam")
    work_root = os.environ.get("QDOCK_QPU_WORKDIR") or pick_latest_run(base_work_root)
    if not work_root:
        print("No workdir found under %s" % base_work_root)
        sys.exit(1)

    output_root = os.environ.get("QDOCK_EVAL_DIR") or os.path.join(work_root, "D_try_fam")
    os.makedirs(output_root, exist_ok=True)
    sample_root = os.environ.get("QDOCK_SAMPLE_DIR") or output_root

    gasteiger_map, baseline_path, run_config = load_baseline_results(work_root)
    params = run_config if run_config else {}

    case_dirs = [
        os.path.join(work_root, d)
        for d in os.listdir(work_root)
        if os.path.isdir(os.path.join(work_root, d)) and not d.startswith(".")
    ]
    case_dirs = [d for d in case_dirs if os.path.isdir(os.path.join(d, "QUBOs"))]
    case_dirs = filter_case_dirs(sorted(case_dirs))

    report = os.environ.get("QDOCK_REPORT") == "1"
    run_tag = os.environ.get("QDOCK_RUN_TAG")
    output_tag = run_tag or get_run_tag()
    results = []
    for case_dir in case_dirs:
        results.append(run_case(
            case_dir,
            sample_root,
            output_root,
            params,
            gasteiger_map,
            report=report,
            run_tag=run_tag,
            output_tag=output_tag,
        ))

    env_info = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "prody_version": getattr(prody, "__version__", None),
        "source_samples": sample_root,
        "source_results": baseline_path,
        "source_workdir": work_root,
        "output_root": output_root,
        "sample_root": sample_root,
        "run_tag": run_tag,
        "output_tag": output_tag,
    }
    summary = build_summary(results, len(case_dirs), 2.0, env_info, params)

    summary_payload = {"summary": summary, "results": results}
    for case_dir in case_dirs:
        pdb_id = os.path.basename(case_dir)
        _, inferred_tag = resolve_case_root(sample_root, pdb_id, run_tag)
        output_case_tag = inferred_tag or output_tag
        output_case_dir = os.path.join(output_root, pdb_id, output_case_tag)
        os.makedirs(output_case_dir, exist_ok=True)
        with open(os.path.join(output_case_dir, "D_qpu_fam_results.json"), "w") as f:
            json.dump(summary_payload, f, indent=2)
        with open(os.path.join(output_case_dir, "D_qpu_fam_results.txt"), "w") as f:
            json.dump(summary_payload, f, indent=2)

    print("Saved summary per case under: %s" % output_root)
    print("success_rate=%s (threshold=2.0, %s/%s)" % (
        summary["success_rate"], summary["success_count"], summary["success_total"]))


if __name__ == "__main__":
    main()
