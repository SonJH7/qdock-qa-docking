#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
DEFAULT_BASE = ROOT / "qpu_opt_final_fixedemb_v113_opt1_r1to10"


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def safe_float(v, default=math.nan):
    try:
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return int(float(v))
    except Exception:
        return default


def find_latest_manifest(results_dir: Path) -> Path:
    mans = sorted(results_dir.glob("*manifest_*.csv"))
    if not mans:
        raise FileNotFoundError(f"manifest not found under: {results_dir}")
    return mans[-1]


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def derive_run_dir(base_dir: Path, row: dict) -> Path:
    case_name = row["case_name"]
    run_tag = row["run_tag"]
    return base_dir / "samples" / case_name / run_tag


def load_metrics_strict(run_dir: Path, ligand: str) -> dict:
    report_csv = run_dir / f"{ligand}_ligand_report.csv"
    meta_json = run_dir / f"{ligand}_ligand_sampler_meta.json"
    if not meta_json.exists():
        meta_json = run_dir / "QUBOs" / f"{ligand}_ligand_sampler_meta.json"
    if not meta_json.exists():
        raise FileNotFoundError(f"missing sampler_meta: {run_dir}")

    meta = json.loads(meta_json.read_text())
    num_reads = safe_int(meta.get("num_reads", 0), 0)
    if num_reads <= 0:
        raise RuntimeError(f"invalid num_reads in {meta_json}")

    timing = meta.get("timing", {}) or {}
    qpu_access_time_us = safe_float(timing.get("qpu_access_time", math.nan), math.nan)
    qpu_access_time_sec = (qpu_access_time_us * 1e-6) if math.isfinite(qpu_access_time_us) else math.nan
    t_shot = (qpu_access_time_sec / num_reads) if (math.isfinite(qpu_access_time_sec) and num_reads > 0) else math.nan

    n_report_rows = 0
    n_valid = 0
    n_success_valid = 0

    with report_csv.open(newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            n_report_rows += 1
            is_valid = parse_bool(r.get("Feasible", ""))
            is_success = parse_bool(r.get("Success(<=2.0A)", ""))
            if is_valid:
                n_valid += 1
                if is_success:
                    n_success_valid += 1

    validity_rate = n_valid / num_reads
    success_rate = (n_success_valid / n_valid) if n_valid > 0 else math.nan
    quality = n_success_valid / num_reads

    if quality <= 0:
        tts_099 = math.inf
    elif quality >= 1:
        tts_099 = t_shot
    else:
        tts_099 = t_shot * math.log(0.01) / math.log(1.0 - quality)

    return {
        "n_reads": num_reads,
        "n_report_rows": n_report_rows,
        "n_valid": n_valid,
        "n_success_valid": n_success_valid,
        "validity_rate": validity_rate,
        "success_rate": success_rate,
        "quality": quality,
        "qpu_access_time_us": qpu_access_time_us,
        "tts_0p99_sec": tts_099,
        "embedding_cached": bool(meta.get("embedding_cached", False)),
        "chain_strength": safe_float((meta.get("sampler_kwargs", {}) or {}).get("chain_strength", math.nan), math.nan),
        "annealing_time": safe_float((meta.get("sampler_kwargs", {}) or {}).get("annealing_time", math.nan), math.nan),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def aggregate_by_ligand(rows: list[dict]) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        g[r["ligand"]].append(r)

    out = []
    for lig, vals in sorted(g.items()):
        qs = [safe_float(v["quality"]) for v in vals]
        ts = [safe_float(v["tts_0p99_sec"]) for v in vals]
        out.append(
            {
                "ligand": lig,
                "n_repeats": len(vals),
                "quality_mean": float(np.mean(qs)),
                "quality_std": float(np.std(qs, ddof=1)) if len(qs) > 1 else 0.0,
                "quality_min": float(np.min(qs)),
                "quality_max": float(np.max(qs)),
                "tts_0p99_mean_sec": float(np.mean(ts)),
                "tts_0p99_std_sec": float(np.std(ts, ddof=1)) if len(ts) > 1 else 0.0,
                "tts_0p99_min_sec": float(np.min(ts)),
                "tts_0p99_max_sec": float(np.max(ts)),
            }
        )
    return out


def make_boxplots(raw_rows: list[dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    ligands = ["3nq9", "4jsz"]
    labels = [
        "3nq9\n(Kdist=7.5, Kmono=19.0, cs=1.0, at=800)",
        "4jsz\n(Kdist=0.5, Kmono=1.0, cs=0.75, at=800)",
    ]

    q_data = []
    t_data = []
    for lig in ligands:
        rows = [r for r in raw_rows if r["ligand"] == lig]
        q_data.append([safe_float(r["quality"]) for r in rows])
        t_data.append([safe_float(r["tts_0p99_sec"]) for r in rows])

    # Quality
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    bp = ax.boxplot(q_data, tick_labels=labels, showfliers=True, patch_artist=True, widths=0.56)
    colors = ["#4C78A8", "#F58518"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.40)
        patch.set_edgecolor("#2b2b2b")
    for k in ["whiskers", "caps", "medians"]:
        for artist in bp[k]:
            artist.set_color("#2b2b2b")
    ax.set_ylabel("Quality = #(valid ∧ RMSD≤2Å) / N_reads", fontsize=12)
    ax.set_title("QPU-Opt Final: Quality Boxplot (Paper Metric)", fontsize=13, pad=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, 1.03)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_dir / "qpu_opt_final_quality_boxplot_papermetric.png", dpi=260)
    plt.close(fig)

    # TTS
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    bp = ax.boxplot(t_data, tick_labels=labels, showfliers=True, patch_artist=True, widths=0.56)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.40)
        patch.set_edgecolor("#2b2b2b")
    for k in ["whiskers", "caps", "medians"]:
        for artist in bp[k]:
            artist.set_color("#2b2b2b")
    ax.set_ylabel("TTS$_{0.99}$ (sec)", fontsize=12)
    ax.set_title("QPU-Opt Final: TTS$_{0.99}$ Boxplot (Paper Metric)", fontsize=13, pad=10)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25, which="both")
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_dir / "qpu_opt_final_tts0p99_boxplot_papermetric.png", dpi=260)
    plt.close(fig)


# ---------- structure rendering ----------

ELEMENT_COLOR = {
    "H": "#F2F2F2",
    "C": "#31A354",
    "N": "#3182BD",
    "O": "#E6550D",
    "S": "#756BB1",
    "P": "#636363",
}

COV_RAD = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
}


def parse_pdb_atoms(pdb_path: Path) -> list[dict]:
    atoms = []
    with pdb_path.open() as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() or line[12:16].strip()[0]
                elem = elem.upper()
                atom_name = line[12:16].strip()
            except Exception:
                continue
            atoms.append({"x": x, "y": y, "z": z, "elem": elem, "name": atom_name})
    return atoms


def parse_pose_model(pose_pdb: Path, pose_id: int) -> list[dict]:
    atoms = []
    cur_model = None
    with pose_pdb.open() as f:
        for line in f:
            if line.startswith("MODEL"):
                cur_model = safe_int(line.split()[-1], 0)
                continue
            if line.startswith("ENDMDL"):
                if cur_model == pose_id:
                    break
                continue
            if cur_model != pose_id:
                continue
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = line[76:78].strip() or line[12:16].strip()[0]
            elem = elem.upper()
            atom_name = line[12:16].strip()
            atoms.append({"x": x, "y": y, "z": z, "elem": elem, "name": atom_name})
    if not atoms:
        raise RuntimeError(f"pose_id {pose_id} not found in {pose_pdb}")
    return atoms


def infer_bonds(atoms: list[dict]) -> list[tuple[int, int]]:
    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    bonds = []
    for i in range(len(atoms)):
        ei = atoms[i]["elem"]
        ri = COV_RAD.get(ei, 0.75)
        for j in range(i + 1, len(atoms)):
            ej = atoms[j]["elem"]
            rj = COV_RAD.get(ej, 0.75)
            d = np.linalg.norm(coords[i] - coords[j])
            if 0.4 < d <= (ri + rj + 0.45):
                bonds.append((i, j))
    return bonds


def set_axes_equal(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2.0
    r = (maxs - mins).max() / 2.0
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)


def pick_best_pose_id(report_csv: Path) -> int:
    best = None
    with report_csv.open(newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            pose_id = safe_int(r.get("Pose_ID", 0), 0)
            rmsd = safe_float(r.get("RMSD(A)", math.inf), math.inf)
            valid = parse_bool(r.get("Feasible", ""))
            succ = parse_bool(r.get("Success(<=2.0A)", ""))
            if not (valid and succ):
                continue
            key = (rmsd, pose_id)
            if best is None or key < best[0]:
                best = (key, pose_id)
    if best is not None:
        return best[1]
    # fallback to first pose
    return 1


def render_structure(receptor_pdb: Path, pose_pdb: Path, report_csv: Path, out_png: Path, title: str) -> tuple[int, Path]:
    pose_id = pick_best_pose_id(report_csv)
    rec_atoms = parse_pdb_atoms(receptor_pdb)
    lig_atoms = parse_pose_model(pose_pdb, pose_id)
    lig_bonds = infer_bonds(lig_atoms)

    rec_xyz = np.array([[a["x"], a["y"], a["z"]] for a in rec_atoms], dtype=float)
    lig_xyz = np.array([[a["x"], a["y"], a["z"]] for a in lig_atoms], dtype=float)

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # receptor pocket points
    for elem in sorted({a["elem"] for a in rec_atoms}):
        idx = [i for i, a in enumerate(rec_atoms) if a["elem"] == elem]
        c = ELEMENT_COLOR.get(elem, "#9E9E9E")
        ax.scatter(
            rec_xyz[idx, 0], rec_xyz[idx, 1], rec_xyz[idx, 2],
            s=65, c=c, alpha=0.35, edgecolors="none", depthshade=False,
        )

    # ligand bonds
    for i, j in lig_bonds:
        xs = [lig_xyz[i, 0], lig_xyz[j, 0]]
        ys = [lig_xyz[i, 1], lig_xyz[j, 1]]
        zs = [lig_xyz[i, 2], lig_xyz[j, 2]]
        ax.plot(xs, ys, zs, color="#2b2b2b", lw=2.2, alpha=0.95)

    # ligand atoms
    for elem in sorted({a["elem"] for a in lig_atoms}):
        idx = [i for i, a in enumerate(lig_atoms) if a["elem"] == elem]
        c = ELEMENT_COLOR.get(elem, "#9E9E9E")
        ax.scatter(
            lig_xyz[idx, 0], lig_xyz[idx, 1], lig_xyz[idx, 2],
            s=145, c=c, alpha=1.0, edgecolors="#111111", linewidths=0.4, depthshade=False,
        )

    xyz_all = np.vstack([rec_xyz, lig_xyz])
    set_axes_equal(ax, xyz_all)
    ax.view_init(elev=20, azim=-58)

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_axis_off()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=320, facecolor="white")
    plt.close(fig)

    # also export selected ligand model pdb for external renderer use
    pose_model_pdb = out_png.with_suffix("").with_name(out_png.stem + f"_pose{pose_id}.pdb")
    with pose_model_pdb.open("w") as f:
        serial = 1
        for a in lig_atoms:
            f.write(
                f"ATOM  {serial:5d} {a['name'][:4]:>4s} LIG A   1    "
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}  1.00  0.00          {a['elem'][:2]:>2s}\n"
            )
            serial += 1
        f.write("END\n")

    return pose_id, pose_model_pdb


def write_chimerax_script(receptor_pdb: Path, ligand_model_pdb: Path, out_cxc: Path, title: str) -> None:
    script = f"""
open {receptor_pdb}
open {ligand_model_pdb}
color #1 #B3B3B3
style #1 sphere
transparency #1 55
size #1 atomRadius 0.8
style #2 stick
color #2 byelement
size #2 stickRadius 0.20
set bgColor white
lighting soft
view
save {out_cxc.with_suffix('.png')}
""".strip() + "\n"
    out_cxc.write_text(script)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build QPU-opt final strict-metric boxplots + structure renders")
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    base_dir = args.base_dir.resolve()
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.resolve() if args.manifest else find_latest_manifest(results_dir)
    rows = read_manifest(manifest)
    ok_rows = [r for r in rows if r.get("status", "") == "ok"]

    raw = []
    for r in ok_rows:
        ligand = r["ligand"]
        run_dir = derive_run_dir(base_dir, r)
        m = load_metrics_strict(run_dir, ligand)
        raw.append(
            {
                "ligand": ligand,
                "repeat": safe_int(r.get("repeat", 0), 0),
                "case_name": r.get("case_name", ""),
                "setting": r.get("setting", ""),
                "chain_strength": safe_float(r.get("chain_strength", m["chain_strength"]), m["chain_strength"]),
                "annealing_time": safe_float(r.get("annealing_time", m["annealing_time"]), m["annealing_time"]),
                "run_tag": r.get("run_tag", ""),
                "n_reads": m["n_reads"],
                "n_report_rows": m["n_report_rows"],
                "n_valid": m["n_valid"],
                "n_success_valid": m["n_success_valid"],
                "validity_rate": m["validity_rate"],
                "success_rate": m["success_rate"],
                "quality": m["quality"],
                "qpu_access_time_us": m["qpu_access_time_us"],
                "tts_0p99_sec": m["tts_0p99_sec"],
                "embedding_cached": m["embedding_cached"],
            }
        )

    raw.sort(key=lambda x: (x["ligand"], x["repeat"]))
    agg = aggregate_by_ligand(raw)

    raw_csv = results_dir / "qpu_opt_final_quality_tts_raw_papermetric.csv"
    agg_csv = results_dir / "qpu_opt_final_quality_tts_agg_papermetric.csv"

    write_csv(
        raw_csv,
        raw,
        [
            "ligand",
            "repeat",
            "case_name",
            "setting",
            "chain_strength",
            "annealing_time",
            "run_tag",
            "n_reads",
            "n_report_rows",
            "n_valid",
            "n_success_valid",
            "validity_rate",
            "success_rate",
            "quality",
            "qpu_access_time_us",
            "tts_0p99_sec",
            "embedding_cached",
        ],
    )
    write_csv(
        agg_csv,
        agg,
        [
            "ligand",
            "n_repeats",
            "quality_mean",
            "quality_std",
            "quality_min",
            "quality_max",
            "tts_0p99_mean_sec",
            "tts_0p99_std_sec",
            "tts_0p99_min_sec",
            "tts_0p99_max_sec",
        ],
    )

    make_boxplots(raw, figures_dir)

    # Select best run per ligand by quality desc, tts asc
    by_lig = defaultdict(list)
    for r in raw:
        by_lig[r["ligand"]].append(r)

    render_outputs = {}
    for lig in ["3nq9", "4jsz"]:
        vals = by_lig[lig]
        vals.sort(key=lambda x: (-x["quality"], x["tts_0p99_sec"], x["repeat"]))
        best = vals[0]
        run_dir = base_dir / "samples" / best["case_name"] / best["run_tag"]
        pose_pdb = run_dir / "Poses" / f"{lig}_ligand_poses.pdb"
        report_csv = run_dir / f"{lig}_ligand_report.csv"
        receptor_pdb = base_dir / "work_cases" / best["case_name"] / "pocs" / "receptor_fp_001.pdb"
        out_png = figures_dir / f"qpu_opt_final_structure_{lig}_best.png"
        title = f"{lig} best run (repeat={best['repeat']}, quality={best['quality']:.3f})"

        pose_id, ligand_model_pdb = render_structure(
            receptor_pdb=receptor_pdb,
            pose_pdb=pose_pdb,
            report_csv=report_csv,
            out_png=out_png,
            title=title,
        )
        out_cxc = figures_dir / f"qpu_opt_final_structure_{lig}_best_chimerax.cxc"
        write_chimerax_script(receptor_pdb, ligand_model_pdb, out_cxc, title)

        render_outputs[lig] = {
            "png": out_png,
            "pose_id": pose_id,
            "pose_pdb": ligand_model_pdb,
            "cxc": out_cxc,
            "best_repeat": best["repeat"],
            "best_quality": best["quality"],
        }

    # two-panel combined structure figure
    fig = plt.figure(figsize=(12.5, 5.5))
    for i, lig in enumerate(["3nq9", "4jsz"], start=1):
        img = plt.imread(render_outputs[lig]["png"])
        ax = fig.add_subplot(1, 2, i)
        ax.imshow(img)
        ax.set_title(
            f"{lig} (best repeat={render_outputs[lig]['best_repeat']}, pose={render_outputs[lig]['pose_id']})",
            fontsize=11,
        )
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(figures_dir / "qpu_opt_final_structure_best_panel.png", dpi=260)
    plt.close(fig)

    print(f"manifest={manifest}")
    print(f"raw_csv={raw_csv}")
    print(f"agg_csv={agg_csv}")
    print(f"quality_fig={figures_dir / 'qpu_opt_final_quality_boxplot_papermetric.png'}")
    print(f"tts_fig={figures_dir / 'qpu_opt_final_tts0p99_boxplot_papermetric.png'}")
    print(f"structure_panel={figures_dir / 'qpu_opt_final_structure_best_panel.png'}")
    for lig in ["3nq9", "4jsz"]:
        d = render_outputs[lig]
        print(f"structure_{lig}={d['png']} (pose_id={d['pose_id']})")
        print(f"chimerax_{lig}={d['cxc']}")


if __name__ == "__main__":
    main()
