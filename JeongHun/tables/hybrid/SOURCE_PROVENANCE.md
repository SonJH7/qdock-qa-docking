# Table 4 (`tab:benchmarks`) source provenance

The following files provide the source values for the cross-solver benchmark
table. Paths are given relative to the repository root.

## Hybrid QA and exact MIP

| Table-4 rows | by-ligand summary (mRMSD/Quality/Validity/TTS) | run_summary (per-run min = Best RMSD) |
|--------------|------------------------------------------------|----------------------------------------|
| NL-hybrid | `JeongHun/tables/hybrid/nl_opt_summary_by_ligand.csv` | `JeongHun/tables/hybrid/nl_opt_run_summary.csv` |
| CQM-hybrid | `JeongHun/tables/hybrid/cqm_opt_summary_by_ligand.csv` | `JeongHun/tables/hybrid/cqm_opt_run_summary.csv` |
| BQM-hybrid | `JeongHun/tables/hybrid/bqm_hybrid_summary_by_ligand.csv` | `JeongHun/tables/hybrid/bqm_hybrid_run_summary.csv` |
| CQM->BQM | `JeongHun/tables/hybrid/cqm_opt_bqm_from_cqm_summary_by_ligand.csv` | `JeongHun/tables/hybrid/cqm_opt_bqm_from_cqm_run_summary.csv` |
| Gurobi / CPLEX (opt) | `benchmarks/classic/exact_opt_pose_metrics.csv` | (single run: Best = mRMSD) |

The hybrid rows were run at the penalty pair recorded in each `run_summary.csv`
(`Kdist`/`Kmono` columns), which is not the QA operating point `kappa*`. The
hybrid and QA rows are therefore not run at matched penalties.

## Simulated annealing

The SA rows are sourced from `JeongHun/tables/sa_penalty/sa_penalty_summary.csv`.
Table 4 reports the highest-Quality penalty cell for each target:

- 3NQ9 (Kdist 2.0, Kmono 3.0): mRMSD 0.737, Quality 0.066, Validity 1.000, TTS 160.819, runtime 2.384.
- 4JSZ (Kdist 1.0, Kmono 10.0): mRMSD 0.992, Quality 0.116, Validity 1.000, TTS 88.683, runtime 2.374.

These cells are reproduced by taking the row with the maximum
`success_rate_mean` per ligand from that file.

## QDock and AutoDock Vina

- QDock BO-QA -> `JeongHun/tables/final_bo_qa/quality_tts_agg.csv`
  (per-run rows in `quality_tts_raw.csv`).
- AutoDock Vina (v1.2.5) -> `vina_3nq9_run/vina_3nq9_summary.csv` and
  `vina_4jsz_run/vina_4jsz_summary.csv`, with per-mode rows in the
  corresponding `vina_*_raw.csv` and the run scripts in
  `JeongHun/tables/vina/`.

## Frozen-embedding statistics

`JeongHun/tables/embedding/embedding_stats.csv` records logical and physical
problem sizes, chain statistics, selected parameters, and the SHA-256 digests of
the frozen embedding maps and of the QUBO instances actually submitted.
