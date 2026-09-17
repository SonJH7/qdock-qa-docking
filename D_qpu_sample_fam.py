#!/usr/bin/env python3
import os
import sys
import json
import time
import statistics
import shutil
from datetime import datetime

import numpy as np


def summarize_times(values):
    if not values:
        return {"avg": None, "median": None, "max": None, "total": None}
    return {
        "avg": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
        "total": round(sum(values), 3),
    }


def load_sampler_from_env():
    cfg_path = os.environ.get("QDOCK_SAMPLER_CONFIG")
    if cfg_path:
        from QDock.samplers.D_factory import SamplerFactory, load_config
        cfg = load_config(cfg_path)
        sampler = SamplerFactory.from_config(cfg)
        sampler_kwargs = cfg.get("sampler_kwargs", {}) or {}
        if "num_reads" in cfg and "num_reads" not in sampler_kwargs:
            sampler_kwargs["num_reads"] = cfg["num_reads"]
        return sampler, sampler_kwargs, cfg_path, cfg
    from QDock.samplers.D_dwave_qpu import DWaveQPUSamplerAdapter
    return DWaveQPUSamplerAdapter(), {}, None, {}


def load_qubo(path):
    return np.load(path, allow_pickle=True).item()


def load_embedding(path):
    with open(path, "r") as f:
        return json.load(f)


def save_embedding(path, embedding):
    with open(path, "w") as f:
        json.dump(embedding, f, indent=2, sort_keys=True, default=str)


def save_sampler_meta(path, meta):
    try:
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True, default=str)
    except Exception:
        pass


def save_samples(path, samples, energies, chain_break_fractions=None):
    payload = {
        "samples": samples,
        "energies": energies,
    }
    if chain_break_fractions is not None:
        payload["chain_break_fraction"] = chain_break_fractions
    with open(path, "w") as f:
        json.dump(payload, f)


def save_text_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


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


def filter_case_dirs(case_dirs):
    ids_env = os.environ.get("QDOCK_PDB_IDS")
    if ids_env:
        wanted = {item.strip().lower() for item in ids_env.split(",") if item.strip()}
        case_dirs = [d for d in case_dirs if os.path.basename(d).lower() in wanted]
    return case_dirs


def apply_limit(case_dirs):
    limit_env = os.environ.get("QDOCK_LIMIT")
    if limit_env:
        try:
            limit_val = int(limit_env)
        except Exception:
            limit_val = None
        if limit_val is not None:
            case_dirs = case_dirs[:limit_val]
    return case_dirs


def load_qubits_map(results_path):
    if not results_path or not os.path.exists(results_path):
        return {}
    try:
        data = json.load(open(results_path))
        qubits_map = {}
        for item in data.get("results", []):
            pdb_id = item.get("pdb_id")
            qubits = item.get("qubits")
            if pdb_id and qubits is not None:
                qubits_map[pdb_id.lower()] = qubits
        return qubits_map
    except Exception:
        return {}


def sort_case_dirs_by_qubits(case_dirs, results_path):
    qubits_map = load_qubits_map(results_path)

    def sort_key(path):
        pdb_id = os.path.basename(path)
        qubits = qubits_map.get(pdb_id.lower())
        if qubits is None:
            qubits = 10**9
        return (qubits, pdb_id)

    return sorted(case_dirs, key=sort_key)

def load_run_config(work_root, default_params):
    results_path = os.path.join(work_root, "try_all_sa_fam_results.json")
    if not os.path.exists(results_path):
        return default_params, None
    try:
        data = json.load(open(results_path))
        run_cfg = data.get("summary", {}).get("run_config", {})
        params = dict(default_params)
        for key in params:
            if key in run_cfg and run_cfg[key] is not None:
                params[key] = run_cfg[key]
        return params, results_path
    except Exception:
        return default_params, None


def extract_var_count(qubo):
    vars_set = set()
    for u, v in qubo.keys():
        vars_set.add(u)
        vars_set.add(v)
    return len(vars_set)

def find_latest_embedding(output_root, pdb_id, ligand_name, exclude_run_tag=None):
    case_root = os.path.join(output_root, pdb_id)
    if not os.path.isdir(case_root):
        return None
    best_path = None
    best_mtime = None
    for run_tag in os.listdir(case_root):
        if run_tag.startswith("."):
            continue
        if exclude_run_tag and run_tag == exclude_run_tag:
            continue
        run_dir = os.path.join(case_root, run_tag, "QUBOs")
        if not os.path.isdir(run_dir):
            continue
        cand = os.path.join(run_dir, "%s_embedding.json" % ligand_name)
        if not os.path.exists(cand):
            continue
        mtime = os.path.getmtime(cand)
        if best_mtime is None or mtime > best_mtime:
            best_mtime = mtime
            best_path = cand
    return best_path


def is_sa_sampler(cfg, sampler):
    backend = None
    if isinstance(cfg, dict):
        backend = cfg.get("backend")
    if backend == "neal":
        return True
    return type(sampler).__name__.lower().startswith("neal")


def parse_int_env(name):
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def count_completed_runs(case_output_root):
    if not os.path.isdir(case_output_root):
        return 0
    count = 0
    for name in os.listdir(case_output_root):
        run_dir = os.path.join(case_output_root, name)
        if not os.path.isdir(run_dir) or name.startswith("."):
            continue
        if os.path.exists(os.path.join(run_dir, "D_qpu_fam_samples.json")):
            count += 1
    return count


def run_case(source_case_dir, output_root, run_tag, params, sampler, sampler_kwargs):
    start = time.perf_counter()
    start_cpu = time.process_time()
    load_sec = None
    sample_sec = None
    sample_cpu_sec = None
    io_sec = None
    pdb_id = os.path.basename(source_case_dir)
    qubo_dir = os.path.join(source_case_dir, "QUBOs")
    output_case_dir = os.path.join(output_root, pdb_id, run_tag)
    output_qubo_dir = os.path.join(output_case_dir, "QUBOs")
    result = {
        "pdb_id": pdb_id,
        "run_tag": run_tag,
        "params": dict(params),
        "paths": {
            "source_dir": source_case_dir,
            "output_dir": output_case_dir,
            "source_qubo_dir": qubo_dir,
            "output_qubo_dir": output_qubo_dir,
        },
    }
    os.makedirs(output_case_dir, exist_ok=True)
    os.makedirs(output_qubo_dir, exist_ok=True)
    try:
        if not os.path.isdir(qubo_dir):
            result.update({"status": "error", "error_type": "missing_qubo", "error_msg": "QUBOs dir not found"})
            return result
        qubo_files = [f for f in os.listdir(qubo_dir) if f.endswith(".npy")]
        if not qubo_files:
            result.update({"status": "error", "error_type": "missing_qubo", "error_msg": "QUBO file not found"})
            return result

        qubo_file = sorted(qubo_files)[0]
        ligand_name = qubo_file[:-4]
        qubo_path = os.path.join(qubo_dir, qubo_file)
        result["paths"]["qubo_file"] = qubo_path

        load_start = time.perf_counter()
        qubo = load_qubo(qubo_path)
        load_sec = time.perf_counter() - load_start
        qubits = extract_var_count(qubo)
        estimated_terms = int(qubits * (qubits - 1) / 2)

        sampler_kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)
        num_reads = sampler_kwargs.pop("num_reads", params.get("n_pos"))
        embedding_path = os.path.join(output_qubo_dir, "%s_embedding.json" % ligand_name)
        embedding_reused = False
        embedding_source = None
        reuse_flag = os.environ.get("QDOCK_EMBEDDING_REUSE") == "1"
        if reuse_flag and not os.path.exists(embedding_path):
            latest = find_latest_embedding(output_root, pdb_id, ligand_name, exclude_run_tag=run_tag)
            if latest:
                try:
                    shutil.copy2(latest, embedding_path)
                    embedding_reused = True
                    embedding_source = latest
                except Exception:
                    pass
        embedding_cached = False
        if os.path.exists(embedding_path):
            try:
                sampler_kwargs["embedding"] = load_embedding(embedding_path)
                embedding_cached = True
            except Exception:
                sampler_kwargs.pop("embedding", None)
        sample_start = time.perf_counter()
        sample_cpu_start = time.process_time()
        raw_solution = sampler.sample_qubo(qubo, num_reads=num_reads, **sampler_kwargs)
        sample_sec = time.perf_counter() - sample_start
        sample_cpu_sec = time.process_time() - sample_cpu_start

        meta = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sampler": type(sampler).__name__,
            "num_reads": num_reads,
            "sampler_kwargs": dict(sampler_kwargs),
            "embedding_cached": embedding_cached,
            "embedding_path": embedding_path,
            "embedding_reused": embedding_reused,
            "embedding_source": embedding_source,
        }
        if hasattr(sampler, "get_meta"):
            meta.update(sampler.get_meta())
        if hasattr(raw_solution, "info"):
            info = raw_solution.info
            if isinstance(info, dict):
                if "problem_id" in info:
                    meta["problem_id"] = info.get("problem_id")
                if "timing" in info:
                    meta["timing"] = info.get("timing")
                embedding = info.get("embedding_context", {}).get("embedding")
                if embedding:
                    save_embedding(embedding_path, embedding)
                    try:
                        meta.setdefault("physical_qubits", sum(len(v) for v in embedding.values()))
                        meta.setdefault("max_chain_length", max(len(v) for v in embedding.values()))
                    except Exception:
                        pass
                meta["info"] = info
        if "physical_qubits" not in meta and embedding_cached and os.path.exists(embedding_path):
            try:
                embedding = load_embedding(embedding_path)
                if embedding:
                    meta.setdefault("physical_qubits", sum(len(v) for v in embedding.values()))
                    meta.setdefault("max_chain_length", max(len(v) for v in embedding.values()))
            except Exception:
                pass
        io_start = time.perf_counter()
        save_sampler_meta(os.path.join(output_qubo_dir, "%s_sampler_meta.json" % ligand_name), meta)
        save_sampler_meta(os.path.join(output_case_dir, "%s_sampler_meta.json" % ligand_name), meta)
        save_text_json(os.path.join(output_case_dir, "%s_sampler_meta.txt" % ligand_name), meta)

        samples = []
        for sample in raw_solution.samples():
            samples.append({k: int(v) for k, v in sample.items()})
        energies = [float(e) for e in raw_solution.record.energy]
        chain_break_fractions = None
        try:
            dtype = getattr(raw_solution.record, "dtype", None)
            names = list(getattr(dtype, "names", []) or [])
            if "chain_break_fraction" in names:
                chain_break_fractions = [float(x) for x in raw_solution.record.chain_break_fraction]
        except Exception:
            chain_break_fractions = None
        if chain_break_fractions is None:
            try:
                dv = getattr(raw_solution, "data_vectors", None)
                if isinstance(dv, dict) and "chain_break_fraction" in dv:
                    chain_break_fractions = [float(x) for x in dv.get("chain_break_fraction", [])]
            except Exception:
                chain_break_fractions = None
        samples_path = os.path.join(output_qubo_dir, "%s_samples.json" % ligand_name)
        save_samples(samples_path, samples, energies, chain_break_fractions)
        save_samples(os.path.join(output_case_dir, "%s_samples.json" % ligand_name), samples, energies, chain_break_fractions)
        save_text_json(os.path.join(output_case_dir, "%s_samples.txt" % ligand_name), {
            "samples": samples,
            "energies": energies,
            "chain_break_fraction": chain_break_fractions,
        })
        io_sec = time.perf_counter() - io_start

        result.update({
            "status": "ok",
            "qubits": qubits,
            "estimated_terms": estimated_terms,
            "n_samples": len(samples),
            "samples_path": samples_path,
            "sampler_meta": meta,
            "problem_id": meta.get("problem_id"),
            "physical_qubits": meta.get("physical_qubits"),
            "max_chain_length": meta.get("max_chain_length"),
            "embedding_reused": embedding_reused,
            "embedding_source": embedding_source,
            "timing": {
                "load_sec": None if load_sec is None else round(load_sec, 3),
                "sample_sec": None if sample_sec is None else round(sample_sec, 3),
                "sample_cpu_sec": None if sample_cpu_sec is None else round(sample_cpu_sec, 3),
                "io_sec": None if io_sec is None else round(io_sec, 3),
            },
        })
        return result
    except Exception as exc:
        result.update({"status": "error", "error_type": "unknown_error", "error_msg": str(exc)})
        return result
    finally:
        result["elapsed_sec"] = round(time.perf_counter() - start, 3)
        result["cpu_sec"] = round(time.process_time() - start_cpu, 3)


def build_summary(results, total_cases, env_info, run_config):
    ok_cases = [r for r in results if r.get("status") == "ok"]
    error_counts = {
        "missing_qubo": 0,
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
    cpu_values = [r["cpu_sec"] for r in results if r.get("cpu_sec") is not None]
    cpu_stats = summarize_times(cpu_values)
    load_values = [
        r.get("timing", {}).get("load_sec")
        for r in results
        if r.get("timing", {}).get("load_sec") is not None
    ]
    load_stats = summarize_times(load_values)
    sample_values = [
        r.get("timing", {}).get("sample_sec")
        for r in results
        if r.get("timing", {}).get("sample_sec") is not None
    ]
    sample_stats = summarize_times(sample_values)
    sample_cpu_values = [
        r.get("timing", {}).get("sample_cpu_sec")
        for r in results
        if r.get("timing", {}).get("sample_cpu_sec") is not None
    ]
    sample_cpu_stats = summarize_times(sample_cpu_values)
    io_values = [
        r.get("timing", {}).get("io_sec")
        for r in results
        if r.get("timing", {}).get("io_sec") is not None
    ]
    io_stats = summarize_times(io_values)

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
        "ok_cases": len(ok_cases),
        "error_counts": error_counts,
        "elapsed_sec_avg": time_stats["avg"],
        "elapsed_sec_median": time_stats["median"],
        "elapsed_sec_max": time_stats["max"],
        "elapsed_sec_total": time_stats["total"],
        "cpu_sec_avg": cpu_stats["avg"],
        "cpu_sec_median": cpu_stats["median"],
        "cpu_sec_max": cpu_stats["max"],
        "cpu_sec_total": cpu_stats["total"],
        "load_sec_avg": load_stats["avg"],
        "load_sec_median": load_stats["median"],
        "load_sec_max": load_stats["max"],
        "load_sec_total": load_stats["total"],
        "sample_sec_avg": sample_stats["avg"],
        "sample_sec_median": sample_stats["median"],
        "sample_sec_max": sample_stats["max"],
        "sample_sec_total": sample_stats["total"],
        "sample_cpu_sec_avg": sample_cpu_stats["avg"],
        "sample_cpu_sec_median": sample_cpu_stats["median"],
        "sample_cpu_sec_max": sample_cpu_stats["max"],
        "sample_cpu_sec_total": sample_cpu_stats["total"],
        "io_sec_avg": io_stats["avg"],
        "io_sec_median": io_stats["median"],
        "io_sec_max": io_stats["max"],
        "io_sec_total": io_stats["total"],
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

    output_root = os.environ.get("QDOCK_SAMPLE_DIR") or os.path.join(work_root, "D_try_fam")
    os.makedirs(output_root, exist_ok=True)

    default_params = {
        "n_pos": 30,
    }
    params, source_results = load_run_config(work_root, default_params)

    sampler, sampler_kwargs, cfg_path, cfg = load_sampler_from_env()
    sa_skip_gte = None
    if is_sa_sampler(cfg, sampler):
        sa_skip_gte = parse_int_env("QDOCK_SA_SKIP_RUNS_GTE")
    qpu_skip_gte = parse_int_env("QDOCK_QPU_SKIP_RUNS_GTE")

    case_dirs = [
        os.path.join(work_root, d)
        for d in os.listdir(work_root)
        if os.path.isdir(os.path.join(work_root, d)) and not d.startswith(".")
    ]
    case_dirs = [d for d in case_dirs if os.path.isdir(os.path.join(d, "QUBOs"))]
    case_dirs = filter_case_dirs(case_dirs)
    case_dirs = sort_case_dirs_by_qubits(case_dirs, source_results)
    case_dirs = apply_limit(case_dirs)

    run_tag = get_run_tag()
    print("run_tag=%s" % run_tag, flush=True)

    results = []
    run_start = time.perf_counter()
    total_cases = len(case_dirs)
    for idx, case_dir in enumerate(case_dirs, 1):
        pdb_id = os.path.basename(case_dir)
        skip_threshold = qpu_skip_gte
        if skip_threshold is None and sa_skip_gte is not None:
            skip_threshold = sa_skip_gte
        if skip_threshold is not None:
            case_output_root = os.path.join(output_root, pdb_id)
            existing_runs = count_completed_runs(case_output_root)
            if existing_runs >= skip_threshold:
                results.append({
                    "pdb_id": pdb_id,
                    "run_tag": run_tag,
                    "status": "skipped",
                    "skip_reason": "runs_gte",
                    "existing_runs": existing_runs,
                    "skip_threshold": skip_threshold,
                })
                elapsed = time.perf_counter() - run_start
                print("[%d/%d] skip  %s existing_runs=%d total=%.1fs" % (
                    idx, total_cases, pdb_id, existing_runs, elapsed
                ), flush=True)
                continue
        elapsed = time.perf_counter() - run_start
        print("[%d/%d] start %s elapsed=%.1fs" % (idx, total_cases, pdb_id, elapsed), flush=True)
        result = run_case(case_dir, output_root, run_tag, params, sampler, sampler_kwargs)
        results.append(result)
        case_elapsed = result.get("elapsed_sec") or 0.0
        elapsed = time.perf_counter() - run_start
        status = result.get("status")
        error_type = result.get("error_type")
        suffix = (" error=%s" % error_type) if error_type else ""
        print("[%d/%d] done  %s status=%s case=%.1fs total=%.1fs%s" % (
            idx, total_cases, pdb_id, status, case_elapsed, elapsed, suffix
        ), flush=True)
        error_msg = result.get("error_msg")
        if error_msg:
            msg = " ".join(str(error_msg).split())
            if len(msg) > 200:
                msg = msg[:197] + "..."
            print("  error_msg=%s" % msg, flush=True)

    env_info = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "sampler_config": cfg_path,
        "source_results": source_results,
        "source_workdir": work_root,
        "output_root": output_root,
        "run_tag": run_tag,
    }
    summary = build_summary(results, len(case_dirs), env_info, params)

    summary_payload = {"summary": summary, "results": results}
    for case_dir in case_dirs:
        output_case_dir = os.path.join(output_root, os.path.basename(case_dir), run_tag)
        os.makedirs(output_case_dir, exist_ok=True)
        with open(os.path.join(output_case_dir, "D_qpu_fam_samples.json"), "w") as f:
            json.dump(summary_payload, f, indent=2)
        save_text_json(os.path.join(output_case_dir, "D_qpu_fam_samples.txt"), summary_payload)

    print("Saved summary per case under: %s" % output_root)
    print("ok_cases=%s/%s" % (summary["ok_cases"], summary["total_cases"]))


if __name__ == "__main__":
    main()
