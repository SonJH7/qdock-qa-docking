#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import prody


BASE_DIR = os.environ.get("QDOCK_QPU_WORKDIR", "qa_sweeps/1.penalty_test")
SAMPLE_ROOT = os.environ.get("QDOCK_SAMPLE_DIR", os.path.join(BASE_DIR, "D_try_fam"))
RUN_TAG = os.environ.get("QDOCK_RUN_TAG")
OUT_DIR = os.environ.get("QDOCK_OUT_DIR", "penalty_result/qpu_energy_vs_rmsd")
MAX_POINTS = int(os.environ.get("QDOCK_MAX_POINTS", "200000"))
RNG_SEED = int(os.environ.get("QDOCK_RNG_SEED", "42"))


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


def list_case_dirs(root_dir):
    cases = []
    if not os.path.isdir(root_dir):
        return cases
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        # Only include actual K-sweep cases
        if "Kdist" not in name or "Kmono" not in name:
            continue
        cases.append(path)
    return cases


def infer_common_run_tag(sample_root, case_dirs):
    counts = Counter()
    for case_dir in case_dirs:
        subdir = os.path.join(sample_root, os.path.basename(case_dir))
        if not os.path.isdir(subdir):
            continue
        for name in os.listdir(subdir):
            path = os.path.join(subdir, name)
            if os.path.isdir(path):
                counts[name] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def resolve_case_sample_dir(sample_root, case_name, run_tag):
    base = os.path.join(sample_root, case_name)
    if not os.path.isdir(base):
        return None
    if run_tag:
        path = os.path.join(base, run_tag)
        return path if os.path.isdir(path) else None
    # fallback to latest by mtime
    candidates = [
        os.path.join(base, name)
        for name in os.listdir(base)
        if os.path.isdir(os.path.join(base, name))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_ligand_id(case_name):
    return case_name.split("_", 1)[0]


def calc_rmsd(coords_a, coords_b):
    diff = coords_a - coords_b
    return float(np.sqrt((diff * diff).sum() / len(coords_a)))


def load_samples(samples_path):
    with open(samples_path) as f:
        data = json.load(f)
    samples = data.get("samples") or []
    energies = data.get("energies") or []
    return samples, energies


def analyze_case(case_dir, sample_dir, run_tag):
    case_name = os.path.basename(case_dir)
    ligand_id = parse_ligand_id(case_name)
    ligand_name = f"{ligand_id}_ligand"

    ligand_pdb = os.path.join(case_dir, "Ligands", f"{ligand_name}.pdb")
    ligand_pdbqt = os.path.join(case_dir, "Ligands", f"{ligand_name}.pdbqt")
    pocs_dir = os.path.join(case_dir, "pocs")

    if not os.path.isdir(pocs_dir):
        return None, f"missing_pocs:{case_name}"
    if not os.path.exists(ligand_pdb) and not os.path.exists(ligand_pdbqt):
        return None, f"missing_ligand:{case_name}"

    feature_atoms = build_feature_atoms(pocs_dir)
    if feature_atoms is None:
        return None, f"missing_feature_atoms:{case_name}"
    feature_coords = feature_atoms.getCoords()

    ligand_struct = prody.parsePDB(ligand_pdb) if os.path.exists(ligand_pdb) else None
    if ligand_struct is None:
        ligand_struct = prody.parsePDB(ligand_pdbqt)
    ligand_coords = ligand_struct.getCoords()

    sample_path = os.path.join(sample_dir, f"{ligand_name}_samples.json")
    if not os.path.exists(sample_path):
        return None, f"missing_samples:{case_name}"

    samples, energies = load_samples(sample_path)
    if not samples or not energies:
        return None, f"empty_samples:{case_name}"
    if len(samples) != len(energies):
        return None, f"mismatch_samples:{case_name}"

    min_energy = min(energies)
    rmsds = []
    edeltas = []
    valid_rmsds = []
    valid_edeltas = []
    success_flags = []
    valid_flags = []
    valid_rmsds = []
    valid_edeltas = []

    for sample, energy in zip(samples, energies):
        ls = []
        gs = []
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
        pose = prody.applyTransformation(trans, ligand_coords)
        rmsd_val = calc_rmsd(pose, ligand_coords)
        rmsds.append(rmsd_val)
        edelta_val = float(energy - min_energy)
        edeltas.append(edelta_val)
        try:
            res = prody.superpose(ligand_coords, pose)
            mapped_orig = res[0]
            shape_diff = prody.calcRMSD(mapped_orig, pose)
        except Exception:
            shape_diff = float("inf")
        is_valid = shape_diff < 0.1
        valid_flags.append(is_valid)
        is_success = rmsd_val <= 2.0
        success_flags.append(is_success)
        if is_valid:
            valid_rmsds.append(rmsd_val)
            valid_edeltas.append(edelta_val)

    # Top-10% success metrics (per case)
    top_frac = 0.10
    top_n = max(1, int(math.ceil(len(edeltas) * top_frac))) if edeltas else 0
    if edeltas:
        top_idx = sorted(range(len(edeltas)), key=lambda i: edeltas[i])[:top_n]
        top_success_any = any(success_flags[i] for i in top_idx)
        top_success_rate = sum(1 for i in top_idx if success_flags[i]) / float(top_n)
    else:
        top_success_any = None
        top_success_rate = None

    valid_idx = [i for i, v in enumerate(valid_flags) if v]
    if valid_idx:
        valid_idx_sorted = sorted(valid_idx, key=lambda i: edeltas[i])
        top_n_valid = max(1, int(math.ceil(len(valid_idx_sorted) * top_frac)))
        top_valid_idx = valid_idx_sorted[:top_n_valid]
        top_success_any_valid = any(success_flags[i] for i in top_valid_idx)
        top_success_rate_valid = sum(1 for i in top_valid_idx if success_flags[i]) / float(top_n_valid)
    else:
        top_n_valid = 0
        top_success_any_valid = None
        top_success_rate_valid = None

    return {
        "ligand": ligand_id,
        "case": case_name,
        "run_tag": os.path.basename(sample_dir),
        "samples_total": len(samples),
        "samples_used": len(rmsds),
        "energy_min": float(min_energy),
        "rmsds": rmsds,
        "edeltas": edeltas,
        "valid_rmsds": valid_rmsds,
        "valid_edeltas": valid_edeltas,
        "top_frac": top_frac,
        "top_n": top_n,
        "top_success_any": top_success_any,
        "top_success_rate": top_success_rate,
        "top_valid_n": top_n_valid,
        "top_success_any_valid": top_success_any_valid,
        "top_success_rate_valid": top_success_rate_valid,
    }, None


def pearson_corr(x, y):
    if len(x) < 2:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    case_dirs = list_case_dirs(BASE_DIR)
    if not case_dirs:
        raise SystemExit(f"No cases under {BASE_DIR}")

    run_tag = RUN_TAG or infer_common_run_tag(SAMPLE_ROOT, case_dirs)
    if not run_tag:
        raise SystemExit("Could not infer run_tag. Set QDOCK_RUN_TAG.")

    errors = []
    per_case = []
    all_by_ligand = {
        "3nq9": {"rmsd": [], "edelta": [], "valid_rmsd": [], "valid_edelta": []},
        "4jsz": {"rmsd": [], "edelta": [], "valid_rmsd": [], "valid_edelta": []},
    }

    start = time.time()
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        sample_dir = resolve_case_sample_dir(SAMPLE_ROOT, case_name, run_tag)
        if not sample_dir:
            errors.append(f"missing_sample_dir:{case_name}")
            continue
        result, err = analyze_case(case_dir, sample_dir, run_tag)
        if err:
            errors.append(err)
            continue
        per_case.append(result)
        ligand_id = result["ligand"]
        if ligand_id in all_by_ligand:
            all_by_ligand[ligand_id]["rmsd"].extend(result["rmsds"])
            all_by_ligand[ligand_id]["edelta"].extend(result["edeltas"])
            all_by_ligand[ligand_id]["valid_rmsd"].extend(result["valid_rmsds"])
            all_by_ligand[ligand_id]["valid_edelta"].extend(result["valid_edeltas"])

    duration = time.time() - start

    # Summary CSV (all samples)
    summary_csv = os.path.join(OUT_DIR, "energy_vs_rmsd_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ligand",
            "cases",
            "samples_total",
            "samples_used",
            "pearson_r",
            "rmsd_mean",
            "rmsd_std",
            "edelta_mean",
            "edelta_std",
        ])
        for ligand_id, data in all_by_ligand.items():
            rmsd = data["rmsd"]
            edelta = data["edelta"]
            if not rmsd:
                continue
            writer.writerow([
                ligand_id,
                sum(1 for r in per_case if r["ligand"] == ligand_id),
                sum(r["samples_total"] for r in per_case if r["ligand"] == ligand_id),
                len(rmsd),
                pearson_corr(edelta, rmsd),
                float(np.mean(rmsd)),
                float(np.std(rmsd)),
                float(np.mean(edelta)),
                float(np.std(edelta)),
            ])

    # Summary CSV (valid-only)
    valid_csv = os.path.join(OUT_DIR, "energy_vs_rmsd_valid_summary.csv")
    with open(valid_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ligand",
            "cases",
            "samples_total",
            "samples_used",
            "valid_samples_used",
            "pearson_r",
            "rmsd_mean",
            "rmsd_std",
            "edelta_mean",
            "edelta_std",
        ])
        for ligand_id, data in all_by_ligand.items():
            rmsd = data["valid_rmsd"]
            edelta = data["valid_edelta"]
            if not rmsd:
                continue
            writer.writerow([
                ligand_id,
                sum(1 for r in per_case if r["ligand"] == ligand_id),
                sum(r["samples_total"] for r in per_case if r["ligand"] == ligand_id),
                sum(r["samples_used"] for r in per_case if r["ligand"] == ligand_id),
                len(rmsd),
                pearson_corr(edelta, rmsd),
                float(np.mean(rmsd)),
                float(np.std(rmsd)),
                float(np.mean(edelta)),
                float(np.std(edelta)),
            ])

    # Top-10% success per case
    top_case_csv = os.path.join(OUT_DIR, "energy_top10_success_cases.csv")
    with open(top_case_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ligand",
            "case",
            "run_tag",
            "top_frac",
            "top_n",
            "top_success_any",
            "top_success_rate",
            "top_valid_n",
            "top_success_any_valid",
            "top_success_rate_valid",
        ])
        for r in per_case:
            writer.writerow([
                r["ligand"],
                r["case"],
                r["run_tag"],
                r["top_frac"],
                r["top_n"],
                int(r["top_success_any"]) if r["top_success_any"] is not None else "",
                r["top_success_rate"] if r["top_success_rate"] is not None else "",
                r["top_valid_n"],
                int(r["top_success_any_valid"]) if r["top_success_any_valid"] is not None else "",
                r["top_success_rate_valid"] if r["top_success_rate_valid"] is not None else "",
            ])

    top_summary_csv = os.path.join(OUT_DIR, "energy_top10_success_summary.csv")
    with open(top_summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ligand",
            "cases",
            "top_success_any_mean",
            "top_success_rate_mean",
            "top_success_any_valid_mean",
            "top_success_rate_valid_mean",
        ])
        for ligand_id in ["3nq9", "4jsz"]:
            rows = [r for r in per_case if r["ligand"] == ligand_id and r["top_success_any"] is not None]
            if not rows:
                continue
            any_mean = sum(1 for r in rows if r["top_success_any"]) / float(len(rows))
            rate_mean = sum(r["top_success_rate"] for r in rows if r["top_success_rate"] is not None) / float(len(rows))
            any_valid_mean = sum(1 for r in rows if r["top_success_any_valid"]) / float(len(rows))
            rate_valid_mean = sum(r["top_success_rate_valid"] for r in rows if r["top_success_rate_valid"] is not None) / float(len(rows))
            writer.writerow([
                ligand_id,
                len(rows),
                any_mean,
                rate_mean,
                any_valid_mean,
                rate_valid_mean,
            ])

    # Scatter plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(RNG_SEED)

    def downsample(x, y, max_points):
        if len(x) <= max_points:
            return x, y
        idx = rng.sample(range(len(x)), max_points)
        return [x[i] for i in idx], [y[i] for i in idx]

    def plot_scatter(ligand_id, x, y, out_path):
        x, y = downsample(x, y, MAX_POINTS)
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, s=4, alpha=0.25, linewidths=0)
        plt.xlabel("Energy delta (per-case, relative)")
        plt.ylabel("RMSD (A)")
        plt.title(f"Energy vs RMSD (QPU) - {ligand_id}\nN={len(x)}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    for ligand_id, data in all_by_ligand.items():
        if data["rmsd"]:
            out_path = os.path.join(OUT_DIR, f"energy_vs_rmsd_{ligand_id}.png")
            plot_scatter(ligand_id, data["edelta"], data["rmsd"], out_path)
        if data["valid_rmsd"]:
            out_path = os.path.join(OUT_DIR, f"energy_vs_rmsd_valid_{ligand_id}.png")
            plot_scatter(ligand_id, data["valid_edelta"], data["valid_rmsd"], out_path)

    # Combined plot
    combined_x = []
    combined_y = []
    combined_valid_x = []
    combined_valid_y = []
    for data in all_by_ligand.values():
        combined_x.extend(data["edelta"])
        combined_y.extend(data["rmsd"])
        combined_valid_x.extend(data["valid_edelta"])
        combined_valid_y.extend(data["valid_rmsd"])
    if combined_x:
        out_path = os.path.join(OUT_DIR, "energy_vs_rmsd_combined.png")
        plot_scatter("combined", combined_x, combined_y, out_path)
    if combined_valid_x:
        out_path = os.path.join(OUT_DIR, "energy_vs_rmsd_valid_combined.png")
        plot_scatter("combined_valid", combined_valid_x, combined_valid_y, out_path)

    # Summary markdown
    summary_md = os.path.join(OUT_DIR, "energy_vs_rmsd_summary.md")
    with open(summary_md, "w") as f:
        f.write("Energy vs RMSD (QPU) summary\n")
        f.write("=================================\n\n")
        f.write(f"Base dir: {BASE_DIR}\n")
        f.write(f"Sample root: {SAMPLE_ROOT}\n")
        f.write(f"Run tag: {run_tag}\n")
        f.write(f"Cases processed: {len(per_case)}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write(f"Elapsed sec: {duration:.1f}\n\n")
        f.write("Feasible definition: shape_diff < 0.1 (same as report CSV)\n")
        f.write("Top-10% metric: success within lowest 10% energy samples (per case)\n\n")
        if errors:
            f.write("Errors (first 20):\n")
            for err in errors[:20]:
                f.write(f"- {err}\n")

    print(f"Done. Cases: {len(per_case)} | Errors: {len(errors)} | Run tag: {run_tag}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
