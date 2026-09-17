#!/usr/bin/env python3
import os
import sys
import json
import shutil
from datetime import datetime


def parse_list(env_value, cast=float):
    if not env_value:
        return None
    items = []
    for part in env_value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items or None


def ensure_copytree(src, dst):
    if not os.path.isdir(src):
        return False
    if os.path.isdir(dst):
        return True
    shutil.copytree(src, dst)
    return True


def ensure_copy(src, dst):
    if not os.path.exists(src):
        return False
    if os.path.exists(dst):
        return True
    shutil.copy2(src, dst)
    return True


def load_ids(default_ids):
    ids_env = os.environ.get("QDOCK_PDB_IDS")
    if not ids_env:
        return default_ids
    return [item.strip().lower() for item in ids_env.split(",") if item.strip()]


def setup_preproc(pdb_id, base_dir, preproc_root, params):
    preproc_dir = os.path.join(preproc_root, pdb_id)
    os.makedirs(preproc_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(preproc_dir)

    receptor_path = os.path.join(base_dir, "data", f"{pdb_id}_protein.pdb")
    ligand_name = f"{pdb_id}_ligand.mol2"
    ligand_path = os.path.join(base_dir, "data", ligand_name)

    if not os.path.exists(receptor_path):
        os.chdir(prev_cwd)
        return None, "missing_receptor"
    if not os.path.exists(ligand_path):
        os.chdir(prev_cwd)
        return None, "missing_ligand"

    from QDock.FeatureAtomMatching.qdock import FAMDock

    fam = FAMDock()
    fam.make_receptor(receptor_path)
    if not os.path.exists(ligand_name):
        shutil.copy2(ligand_path, ligand_name)
    fam.make_ligand([ligand_name])
    fam.make_box_ligand(ligand_name, grid_length=params["grid_length"])

    os.chdir(prev_cwd)
    return {
        "fam": fam,
        "preproc_dir": preproc_dir,
        "ligand_name": ligand_name,
    }, None


def write_case_meta(case_dir, meta):
    path = os.path.join(case_dir, "sweep_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def run_case_combo(fam, preproc_dir, ligand_name, case_dir, params):
    os.makedirs(case_dir, exist_ok=True)

    # Copy required files so evaluation works later.
    ensure_copytree(os.path.join(preproc_dir, "pocs"), os.path.join(case_dir, "pocs"))
    ensure_copytree(os.path.join(preproc_dir, "Ligands"), os.path.join(case_dir, "Ligands"))
    ensure_copy(os.path.join(preproc_dir, "receptor.pdbqt"), os.path.join(case_dir, "receptor.pdbqt"))

    prev_cwd = os.getcwd()
    os.chdir(case_dir)
    try:
        fam.indiv_dock(
            ligand=fam.ligands[0],
            edge_cutoff=params["edge_cutoff"],
            K_dist=params["K_dist"],
            K_mono=params["K_mono"],
            n_pos=1,
            save_qubo=True,
            sim_dock=False,
            save_match=False,
            save_pose=False,
        )
    finally:
        os.chdir(prev_cwd)

    qubo_path = os.path.join(case_dir, "QUBOs", f"{ligand_name[:-5]}.npy")
    return qubo_path


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    adfr_bin = os.environ.get("QDOCK_ADFR_BIN")
    if adfr_bin:
        os.environ["PATH"] = adfr_bin + os.pathsep + os.environ.get("PATH", "")

    sweep_root = os.environ.get("QDOCK_SWEEP_ROOT") or os.path.join(
        base_dir, "qa_sweeps", "1.penalty_test"
    )
    preproc_root = os.path.join(sweep_root, "_preproc")
    os.makedirs(preproc_root, exist_ok=True)
    os.makedirs(sweep_root, exist_ok=True)

    k_dist_values = parse_list(os.environ.get("QDOCK_KDIST_LIST"))
    k_mono_values = parse_list(os.environ.get("QDOCK_KMONO_LIST"))
    if not k_dist_values:
        k_dist_values = [1.5, 2.0, 2.5, 3.0]
    if not k_mono_values:
        k_mono_values = [8.0, 10.0, 12.0, 14.0]

    pdb_ids = load_ids(["3nq9", "4jsz"])

    base_params = {
        "edge_cutoff": 1.87,
        "grid_length": 1.0,
    }

    results = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(sweep_root, f"penalty_sweep_summary_{run_id}.json")

    for pdb_id in pdb_ids:
        prep, err = setup_preproc(pdb_id, base_dir, preproc_root, base_params)
        if err:
            results.append({"pdb_id": pdb_id, "status": err})
            continue

        fam = prep["fam"]
        preproc_dir = prep["preproc_dir"]
        ligand_name = prep["ligand_name"]

        for k_dist in k_dist_values:
            for k_mono in k_mono_values:
                case_name = f"{pdb_id}_Kdist{k_dist}_Kmono{k_mono}"
                case_dir = os.path.join(sweep_root, case_name)
                params = {
                    "edge_cutoff": base_params["edge_cutoff"],
                    "grid_length": base_params["grid_length"],
                    "K_dist": float(k_dist),
                    "K_mono": float(k_mono),
                }
                try:
                    qubo_path = run_case_combo(fam, preproc_dir, ligand_name, case_dir, params)
                    write_case_meta(
                        case_dir,
                        {
                            "pdb_id": pdb_id,
                            "case_name": case_name,
                            "K_dist": params["K_dist"],
                            "K_mono": params["K_mono"],
                            "edge_cutoff": params["edge_cutoff"],
                            "grid_length": params["grid_length"],
                            "preproc_dir": preproc_dir,
                            "qubo_path": qubo_path,
                        },
                    )
                    results.append(
                        {
                            "pdb_id": pdb_id,
                            "case_name": case_name,
                            "status": "ok",
                            "qubo_path": qubo_path,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "pdb_id": pdb_id,
                            "case_name": case_name,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

                with open(summary_path, "w") as f:
                    json.dump({"results": results}, f, indent=2)

    print(f"Saved summary: {summary_path}")
    print("Sweep root:", sweep_root)


if __name__ == "__main__":
    main()
