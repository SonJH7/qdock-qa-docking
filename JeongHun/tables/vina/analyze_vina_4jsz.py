"""Analyze 4JSZ Vina runs: per-pose RMSD vs crystal, per-run best RMSD, Quality (RMSD<=2)."""
import os
import re
import csv
import math
from pathlib import Path

REPO_ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[3]))
).expanduser().resolve()
ROOT = REPO_ROOT / "vina_4jsz_run"
REF_PDB = ROOT / "4jsz_xtal_ref.pdb"

# Parse crystal reference heavy-atom coords (skip H)
def parse_pdb_heavy(path):
    coords = []
    names  = []
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                elem = line[76:78].strip()
                if elem == 'H' or name.startswith('H'):
                    continue
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords.append((x,y,z)); names.append(name)
    return names, coords

# Parse a Vina output pdbqt — yields list of poses; each pose is (affinity, [coords by atom_name])
def parse_vina_pdbqt(path):
    poses = []
    cur_aff = None
    cur_atoms = []  # list of (name, x, y, z)
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                cur_aff = None
                cur_atoms = []
            elif line.startswith("REMARK VINA RESULT:"):
                m = re.search(r"VINA RESULT:\s*([-\d.]+)", line)
                if m:
                    cur_aff = float(m.group(1))
            elif line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                if name.startswith('H'):
                    continue
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                cur_atoms.append((name, x, y, z))
            elif line.startswith("ENDMDL"):
                poses.append((cur_aff, cur_atoms))
    return poses

# Heavy-atom RMSD of two pose lists by name match
def rmsd_by_name(ref_names, ref_coords, pose_atoms):
    pose_dict = {n:(x,y,z) for (n,x,y,z) in pose_atoms}
    sq = 0.0; n = 0
    for nm, (rx,ry,rz) in zip(ref_names, ref_coords):
        if nm in pose_dict:
            x,y,z = pose_dict[nm]
            sq += (x-rx)**2 + (y-ry)**2 + (z-rz)**2
            n += 1
    return math.sqrt(sq/n) if n else float('nan')

ref_names, ref_coords = parse_pdb_heavy(REF_PDB)
print(f"Reference heavy atoms: {len(ref_names)}")

raw_rows = []  # (run, mode, affinity, rmsd)
agg_rows = []  # (run, n_modes, best_rmsd, mean_rmsd, n_success_2A, quality)

for r in range(1, 11):
    path = ROOT/"outputs"/f"vina_4jsz_r{r}.pdbqt"
    poses = parse_vina_pdbqt(path)
    rmsds = []
    affs  = []
    for mode_i, (aff, atoms) in enumerate(poses, 1):
        rm = rmsd_by_name(ref_names, ref_coords, atoms)
        rmsds.append(rm); affs.append(aff)
        raw_rows.append((r, mode_i, aff, rm))
    n_modes = len(rmsds)
    best = min(rmsds) if rmsds else float('nan')
    mean_rmsd = sum(rmsds)/n_modes if rmsds else float('nan')
    n_success = sum(1 for x in rmsds if x <= 2.0)
    quality   = n_success / n_modes if n_modes else 0.0
    agg_rows.append((r, n_modes, best, mean_rmsd, n_success, quality))
    print(f"r{r:>2}: modes={n_modes}, best_RMSD={best:.4f}, mean={mean_rmsd:.4f}, "
          f"#RMSD<=2={n_success}/{n_modes}, quality={quality:.3f}")

# Aggregate across runs
n = len(agg_rows)
best_per_run = [a[2] for a in agg_rows]
mean_per_run = [a[3] for a in agg_rows]
quality_per_run = [a[5] for a in agg_rows]
n_modes_total = agg_rows[0][1]

mRMSD = sum(best_per_run)/n
overall_best = min(best_per_run)
mean_quality = sum(quality_per_run)/n
print("\n=== 4JSZ Vina aggregate ===")
print(f"Runs                : {n}")
print(f"Modes per run        : {n_modes_total}")
print(f"mRMSD (mean of bests): {mRMSD:.4f} Å")
print(f"Best RMSD            : {overall_best:.4f} Å")
print(f"Quality mean         : {mean_quality:.4f}")
print(f"Quality range        : [{min(quality_per_run):.3f}, {max(quality_per_run):.3f}]")

# Save raw + agg CSVs
with open(ROOT/"vina_4jsz_raw.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run","mode","affinity_kcalmol","rmsd_A"])
    for row in raw_rows:
        w.writerow(row)
with open(ROOT/"vina_4jsz_agg.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run","n_modes","best_rmsd_A","mean_rmsd_A","n_success_2A","quality"])
    for row in agg_rows:
        w.writerow(row)
with open(ROOT/"vina_4jsz_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ligand","runs","modes_per_run","mRMSD_A","best_RMSD_A","quality_mean","quality_min","quality_max"])
    w.writerow(["4jsz", n, n_modes_total, round(mRMSD,4), round(overall_best,4),
                round(mean_quality,4), round(min(quality_per_run),3), round(max(quality_per_run),3)])
print(f"\nSaved: {ROOT}/vina_4jsz_raw.csv, vina_4jsz_agg.csv, vina_4jsz_summary.csv")
