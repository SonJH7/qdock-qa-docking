"""Render v1.13 final-recipe ligand overlay (Crystal vs Sampled) with annotations.

Reproduces the design of qpu_opt_overlay_clean_annotated_*.png
(top-left labels "Crystal Structure" / "Sampled Docking Pose",
bottom centered "QPU Opt / PDB ID / mRMSD" caption).
"""
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pymol2

REPO = Path(
    os.environ.get("QDOCK_ROOT", str(Path(__file__).resolve().parents[1]))
).expanduser().resolve()
OUTDIR = REPO / "JeongHun/figures/pymol"

CASES = [
    {
        "ligand": "3nq9",
        "crystal": REPO / "coreset/3nq9/3nq9_ligand.mol2",
        # Reported RQ4a campaign (anneal-time sweep, at=800 us, cs*=1.0);
        # min RMSD 0.6611 A is attained in all ten repetitions.
        "poses_pdb": REPO
        / "annealing_time_sweep_qpu_fixedemb_v113_opt1_3nq9_cs1p0_upto800_r1to10/samples"
        / "3nq9_Kdist7.500_Kmono19.000"
        / "atsweep_3nq9_at800_cs1p0_r1_20260408_023341/Poses/3nq9_ligand_poses.pdb",
        "model": 948,
        "rmsd": 0.6611,
    },
    {
        "ligand": "4jsz",
        "crystal": REPO / "coreset/4jsz/4jsz_ligand.mol2",
        # Reported RQ4a campaign (anneal-time sweep, at=800 us, cs*=0.75);
        # r2 attains the overall best RMSD 1.2445 A across the ten repetitions.
        "poses_pdb": REPO
        / "annealing_time_sweep_qpu_fixedemb_v113_opt1_at800_r1to10/samples"
        / "4jsz_Kdist0.500_Kmono1.000"
        / "atsweep_4jsz_at800_cs0p75_r2_20260407_205920/Poses/4jsz_ligand_poses.pdb",
        "model": 19,
        "rmsd": 1.2445,
    },
]


def render_ligand_overlay(case: dict, out_path: Path) -> None:
    with pymol2.PyMOL() as pm:
        cmd = pm.cmd
        cmd.set("bg_rgb", [1.0, 1.0, 1.0])
        cmd.set("ray_opaque_background", 1)
        cmd.set("antialias", 2)
        cmd.set("specular", 0.2)
        cmd.set("ray_shadows", 0)
        cmd.set("ambient", 0.55)
        cmd.set("orthoscopic", 1)

        cmd.load(str(case["crystal"]), "crys")
        cmd.load(str(case["poses_pdb"]), "samp_all")
        cmd.create("samp", "samp_all", source_state=case["model"], target_state=1)
        cmd.delete("samp_all")
        n_samp = cmd.count_atoms("samp")
        n_crys = cmd.count_atoms("crys")
        print(f"  atoms: crys={n_crys} samp={n_samp}")

        cmd.hide("everything", "all")
        cmd.show("sticks", "crys")
        cmd.show("sticks", "samp")
        cmd.show("spheres", "crys")
        cmd.show("spheres", "samp")
        cmd.color("blue", "crys")
        cmd.color("red", "samp")
        cmd.set("stick_radius", 0.22, "crys")
        cmd.set("stick_radius", 0.22, "samp")
        cmd.set("sphere_scale", 0.32, "crys")
        cmd.set("sphere_scale", 0.32, "samp")

        overlap = 0.6
        cmd.select("ov_c", f"crys within {overlap} of samp")
        cmd.select("ov_s", f"samp within {overlap} of crys")
        cmd.create("ov_obj", "ov_c or ov_s")
        cmd.color("green", "ov_obj")
        cmd.show("spheres", "ov_obj")
        cmd.set("sphere_scale", 0.40, "ov_obj")
        cmd.set("sphere_transparency", 0.10, "ov_obj")

        cmd.orient("crys or samp")
        cmd.zoom("crys or samp", buffer=1.2)

        cmd.png(str(out_path), width=1600, height=1200, dpi=300, ray=1)


def annotate(in_path: Path, out_path: Path, ligand: str, rmsd: float) -> None:
    img = Image.open(in_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    def find_font(size: int) -> ImageFont.ImageFont:
        for name in (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_size = int(H * 0.055)
    cap_size = int(H * 0.05)
    f_title = find_font(title_size)
    f_cap = find_font(cap_size)

    pad = int(W * 0.025)
    draw.text((pad, pad), "Crystal Structure", fill=(0, 0, 200), font=f_title)
    draw.text(
        (pad, pad + int(title_size * 1.15)),
        "Sampled Docking Pose",
        fill=(200, 0, 0),
        font=f_title,
    )

    caption_lines = [
        "QPU Opt",
        f"PDB ID: {ligand.upper()}",
        f"Best RMSD = {rmsd:.3f} Å",
    ]
    line_h = int(cap_size * 1.18)
    total_h = line_h * len(caption_lines)
    y0 = H - int(H * 0.04) - total_h
    for i, line in enumerate(caption_lines):
        bbox = draw.textbbox((0, 0), line, font=f_cap)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        draw.text((x, y0 + i * line_h), line, fill=(0, 0, 0), font=f_cap)

    img.save(out_path)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        raw = OUTDIR / f"qpu_opt_overlay_v113_{case['ligand']}_raw.png"
        final = OUTDIR / f"qpu_opt_overlay_v113_{case['ligand']}.png"
        print(f"[render] {case['ligand']} model={case['model']} rmsd={case['rmsd']}")
        render_ligand_overlay(case, raw)
        annotate(raw, final, case["ligand"], case["rmsd"])
        raw.unlink(missing_ok=True)
        print(f"  -> {final}")


if __name__ == "__main__":
    main()
