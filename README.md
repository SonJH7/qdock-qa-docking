# QDock

This repository contains the QDock feature-atom-matching implementation, the
QPU/SA experiment drivers, and the accompanying figure and table data.

## Repository layout

- `QDock/` contains the FAM formulation and sampler implementations.
- `run_penalty_sweep_fam.py`, `D_qpu_sample_fam.py`, and
  `D_qpu_eval_fam.py` implement the QPU experiment path.
- `D_try_all_sa_fam.py` and `configs/sampler_config_sa.json` implement the SA
  path.
- `BO_Online/scripts/run_online_bo.py` implements online penalty optimization;
  `BO_Online_prereg_20260804/` contains the 20-run BO-versus-random comparison
  (five runs per policy and target).
- `data/` contains the two protein and ligand inputs used in the study.
- `JeongHun/` contains the active figure assets, plotting scripts, and reported
  tables.
- Root-level experiment directories contain the campaign configurations,
  embeddings, manifests, and result summaries used in the study.

## Environment

Create the software environment with:

```bash
conda env create -f environment.yml
conda activate qdock-qst
```

AutoSite/ADFR must be installed separately. Set `QDOCK_ADFR_BIN` to its binary
directory. QPU credentials must be supplied through the local D-Wave Ocean
configuration or `DWAVE_API_TOKEN`; no credential is stored in this repository.

The available experiment entry points can be inspected with:

```bash
python BO_Online/scripts/run_online_bo.py --help
QDOCK_SAMPLER_CONFIG=configs/sampler_config_sa.json python D_try_all_sa_fam.py
```

## Reproducibility data

The repository provides the processed data, configurations, frozen embeddings,
run manifests, and analysis assets supporting the reported figures and tables.
The BO comparison tables under `JeongHun/tables/bo_vs_random/` and the
corresponding Figure 6/7 assets use the same 20-run comparison. The QPU campaign
manifests and aggregates retain their campaign directory names at the repository
root. File integrity can be verified with `DATA_SHA256SUMS`.

## Attribution

The FAM-QUBO encoding and ligand/feature preparation build on the
[QDock implementation by Jinyin Zha](https://github.com/JinyinZha/QDock),
including its electronegativity table and decoder, released under the MIT
License and accompanying Zha et al., *Encoding Molecular Docking for Quantum
Computers* (JCTC 2023, [doi:10.1021/acs.jctc.3c00943](https://doi.org/10.1021/acs.jctc.3c00943)).
The original copyright and permission notice are preserved in [LICENSE](LICENSE).

The original contributions by Jeong-Hun Son, Seon-Geun Jeong, and Won-Joo
Hwang, including the BO loop, sweep drivers, and analysis and plotting
scripts, are also MIT-licensed; see [LICENSE](LICENSE). Third-party code and
data retain their respective notices and terms.
