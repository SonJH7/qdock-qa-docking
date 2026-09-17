# Frozen minor-embedding records

## Frozen embedding maps

The campaign `fixed_embeddings/` directories contain
`3nq9_ligand_embedding.json` and `4jsz_ligand_embedding.json`, the frozen
minor-embedding maps used for the downstream sweeps. Their SHA-256 digests are
`1140c769647c59163eeb1b7013028f78bd4456540d53acee9c1cbc1243df9290` and
`ec859ffbd18b211df17732f99dbe11df31e65f5438fd5e904fa53a1b2cc48349`,
respectively, matching the run manifests.

## `embedding_stats.csv`
This table records `N_logical`, physical-qubit count, maximum and mean chain
length, the chain-length reference strength, selected penalties, coefficient
scale, solver and graph identifiers, and embedding/QUBO digests for each target.

## Final-run manifest

The final campaign's `results/` directory contains the per-run manifest with
the embedding and QUBO digests together with the integrity and completion
status. Every listed final-recipe run reused the frozen embedding
(`hash_ok=true`).

## Software environment

Exact package versions for the sampling and evaluation environments are listed
under `../environment/`.
