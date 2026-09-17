"""Analyze 3NQ9 Vina runs: per-pose RMSD vs crystal, per-run best RMSD, Quality (RMSD<=2)."""
import os
import re, csv, math
from pathlib import Path

REPO_ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[3]))
).expanduser().resolve()
ROOT = REPO_ROOT / "vina_3nq9_run"
REF_PDB = ROOT / "3nq9_xtal_ref.pdb"

def parse_pdb_heavy(path):
    coords, names = [], []
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

def parse_vina_pdbqt(path):
    poses = []
    cur_aff, cur_atoms = None, []
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                cur_aff, cur_atoms = None, []
            elif line.startswith("REMARK VINA RESULT:"):
                m = re.search(r"VINA RESULT:\s*([-\d.]+)", line)
                if m: cur_aff = float(m.group(1))
            elif line.startswith(("ATOM","HETATM")):
                name = line[12:16].strip()
                if name.startswith('H'): continue
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                cur_atoms.append((name, x, y, z))
            elif line.startswith("ENDMDL"):
                poses.append((cur_aff, cur_atoms))
    return poses

def rmsd_by_name(ref_names, ref_coords, pose_atoms):
    pose_dict = {n:(x,y,z) for (n,x,y,z) in pose_atoms}
    sq, n = 0.0, 0
    for nm, (rx,ry,rz) in zip(ref_names, ref_coords):
        if nm in pose_dict:
            x,y,z = pose_dict[nm]
            sq += (x-rx)**2 + (y-ry)**2 + (z-rz)**2; n += 1
    return math.sqrt(sq/n) if n else float('nan')

ref_names, ref_coords = parse_pdb_heavy(REF_PDB)
print(f"Reference heavy atoms: {len(ref_names)}")

raw_rows, agg_rows, mode1_rmsds = [], [], []
for r in range(1, 11):
    poses = parse_vina_pdbqt(ROOT/"outputs"/f"vina_3nq9_r{r}.pdbqt")
    rmsds, affs = [], []
    for mode_i, (aff, atoms) in enumerate(poses, 1):
        rm = rmsd_by_name(ref_names, ref_coords, atoms)
        rmsds.append(rm); affs.append(aff)
        raw_rows.append((r, mode_i, aff, rm))
        if mode_i == 1: mode1_rmsds.append((r, aff, rm))
    n_modes = len(rmsds)
    best = min(rmsds); mean_rmsd = sum(rmsds)/n_modes
    n_success = sum(1 for x in rmsds if x <= 2.0)
    quality = n_success / n_modes
    agg_rows.append((r, n_modes, best, mean_rmsd, n_success, quality))
    print(f"r{r:>2}: best={best:.4f}, mean={mean_rmsd:.4f}, #<=2={n_success}/{n_modes}, Q={quality:.3f}")

n = len(agg_rows)
best_per_run = [a[2] for a in agg_rows]
quality_per_run = [a[5] for a in agg_rows]

print("\n=== 3NQ9 Vina aggregate ===")
print(f"mRMSD = {sum(best_per_run)/n:.4f} A")
print(f"Best  = {min(best_per_run):.4f} A")
print(f"Quality mean = {sum(quality_per_run)/n:.4f}")
print(f"Quality range = [{min(quality_per_run):.3f}, {max(quality_per_run):.3f}]")

print("\nMode 1 (lowest aff) per run:")
for r, aff, rm in mode1_rmsds:
    flag = "<=2" if rm <= 2.0 else ">2"
    print(f"  r{r:>2}: aff={aff:.3f}  rmsd={rm:.4f}  ({flag})")
n_top1 = sum(1 for _,_,rm in mode1_rmsds if rm <= 2.0)
print(f"Top-1 success: {n_top1}/{len(mode1_rmsds)} = {n_top1/len(mode1_rmsds):.3f}")

with open(ROOT/"vina_3nq9_raw.csv","w",newline="") as f:
    csv.writer(f).writerow(["run","mode","affinity_kcalmol","rmsd_A"])
    csv.writer(f).writerows(raw_rows) if False else None
    w = csv.writer(f)
    for row in raw_rows: w.writerow(row)
with open(ROOT/"vina_3nq9_agg.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["run","n_modes","best_rmsd_A","mean_rmsd_A","n_success_2A","quality"])
    for row in agg_rows: w.writerow(row)
with open(ROOT/"vina_3nq9_summary.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["ligand","runs","modes_per_run","mRMSD_A","best_RMSD_A","quality_mean","quality_min","quality_max","top1_success_rate"])
    w.writerow(["3nq9", n, agg_rows[0][1], round(sum(best_per_run)/n,4), round(min(best_per_run),4),
                round(sum(quality_per_run)/n,4), round(min(quality_per_run),3), round(max(quality_per_run),3),
                round(n_top1/len(mode1_rmsds),3)])
print(f"\nSaved CSVs to {ROOT}/")
