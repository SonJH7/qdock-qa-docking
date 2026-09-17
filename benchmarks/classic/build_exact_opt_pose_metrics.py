#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np
import prody


ROOT = Path(__file__).resolve().parents[2]
CLASSIC_DIR = ROOT / "benchmarks" / "classic"
CASES_DIR = ROOT / "qa_sweeps" / "1.penalty_test" / "_cases"
OUT_CSV = CLASSIC_DIR / "exact_opt_pose_metrics.csv"


def _load_feature_coords(case_dir: Path):
    pocs_dir = case_dir / "pocs"
    pocs = []
    for pdb_file in sorted(pocs_dir.glob("*_fp_*.pdb")):
        pocs.append(prody.parsePDB(str(pdb_file)))
    if not pocs:
        raise FileNotFoundError(f"no feature-point PDB files in {pocs_dir}")
    feature_atoms = pocs[0]
    for i in range(1, len(pocs)):
        feature_atoms += pocs[i]
    return feature_atoms.getCoords()


def _load_ligand_coords(case_dir: Path, ligand_name: str):
    lig_pdb = case_dir / "Ligands" / f"{ligand_name}.pdb"
    lig_pdbqt = case_dir / "Ligands" / f"{ligand_name}.pdbqt"
    native_path = lig_pdb if lig_pdb.exists() else lig_pdbqt
    if not native_path.exists():
        raise FileNotFoundError(f"ligand not found for {ligand_name} in {case_dir}")
    native = prody.parsePDB(str(native_path))
    return native.getCoords()


def _decode_pose(ligand_coords, feature_coords, solution_ones):
    ligand_pts = []
    feature_pts = []
    for key in solution_ones:
        parts = key.split("_", 2)
        if len(parts) < 2:
            continue
        i = int(parts[0])
        j = int(parts[1])
        if i < 0 or j < 0:
            continue
        if i >= len(ligand_coords) or j >= len(feature_coords):
            continue
        ligand_pts.append(ligand_coords[i])
        feature_pts.append(feature_coords[j])
    if not ligand_pts:
        return None, 0
    ligand_pts = np.vstack(ligand_pts)
    feature_pts = np.vstack(feature_pts)
    transform = prody.superpose(ligand_pts, feature_pts)[1]
    pose = prody.applyTransformation(transform, ligand_coords)
    return pose, len(ligand_pts)


def evaluate_exact_json(json_path: Path, method_name: str):
    payload = json.loads(json_path.read_text())
    qubo_path = Path(payload["qubo_path"])
    case_name = qubo_path.parent.parent.name
    case_dir = CASES_DIR / case_name
    ligand_name = qubo_path.stem
    target = ligand_name.split("_", 1)[0]

    ligand_coords = _load_ligand_coords(case_dir, ligand_name)
    feature_coords = _load_feature_coords(case_dir)
    pose, n_matches = _decode_pose(
        ligand_coords,
        feature_coords,
        payload.get("solution_ones", []),
    )
    if pose is None:
        raise RuntimeError(f"no active matches in {json_path}")

    rmsd_val = float(prody.calcRMSD(ligand_coords, pose))
    mapped_orig = prody.superpose(ligand_coords, pose)[0]
    shape_diff = float(prody.calcRMSD(mapped_orig, pose))
    validity = 1.0 if shape_diff < 0.1 else 0.0
    quality = 1.0 if rmsd_val <= 2.0 else 0.0

    runtime_sec = float(payload.get("runtime_sec", 0.0))
    if 0.0 < quality < 1.0:
        tts = runtime_sec * math.log(1.0 - 0.99) / math.log(1.0 - quality)
    else:
        tts = None

    return {
        "method": method_name,
        "ligand": target,
        "runs": 1,
        "mRMSD_mean": round(rmsd_val, 3),
        "mRMSD_min": round(rmsd_val, 3),
        "quality": round(quality, 3),
        "validity": round(validity, 3),
        "tts_99_quality_sec": "" if tts is None else round(tts, 3),
        "runtime_mean_sec": round(runtime_sec, 3),
        "shape_diff": shape_diff,
        "n_matches": n_matches,
        "source_json": str(json_path.relative_to(ROOT)),
    }


def main():
    rows = []
    for solver_dir, method_name in [
        (CLASSIC_DIR / "gurobi", "Gurobi (opt)"),
        (CLASSIC_DIR / "cplex", "CPLEX (opt)"),
    ]:
        for json_path in sorted(solver_dir.glob(f"*_{solver_dir.name}.json")):
            rows.append(evaluate_exact_json(json_path, method_name))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "ligand",
        "runs",
        "mRMSD_mean",
        "mRMSD_min",
        "quality",
        "validity",
        "tts_99_quality_sec",
        "runtime_mean_sec",
        "shape_diff",
        "n_matches",
        "source_json",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_CSV}")
    paper_candidates = [
        ROOT / "JungHun" / "tables" / "classic" / "exact_opt_pose_metrics.csv",
        Path.cwd() / "JungHun" / "tables" / "classic" / "exact_opt_pose_metrics.csv",
    ]
    written_paper = set()
    for out_path in paper_candidates:
        out_path = out_path.resolve()
        if out_path in written_paper:
            continue
        jung_dir = out_path.parents[2]
        if not jung_dir.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path}")
        written_paper.add(out_path)
    for row in rows:
        print(
            f"{row['method']} {row['ligand']}: "
            f"mRMSD={row['mRMSD_mean']}, quality={row['quality']}, "
            f"validity={row['validity']}, runtime={row['runtime_mean_sec']}"
        )


if __name__ == "__main__":
    main()
