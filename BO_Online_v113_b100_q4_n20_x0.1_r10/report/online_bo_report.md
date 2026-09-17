# Online BO Report

- mode: `qpu`
- objective: `quality_4jsz`
- run_tag: `online_bo_v113_b100_quality_4jsz_n20_x0p1_r10_chain_20260405_025657`
- budget: `100` (init `20`)
- policy: `bo`
- seed: `20270102`
- xi: `0.1`
- num_reads: `1000`
- sampler_config: `sampler_config_qpu_reads1000_v113_default.json`
- embed_reuse: `False`

## Best observed query

- step: `11`
- (kdist, kmono): `(0.500, 1.000)`
- objective: `0.879000`
- q_3nq9 / q_4jsz: `0.000000` / `0.879000`
- v_3nq9 / v_4jsz: `1.000000` / `1.000000`

## Notes

- This is an online sequential BO run: each point is evaluated before selecting the next point.
- `quality_min` is the manuscript-aligned quality-centric robust objective.
