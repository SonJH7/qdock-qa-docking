# Online BO Report

- mode: `qpu`
- objective: `quality_3nq9`
- run_tag: `online_bo_v113_b100_q3_n20_x0p1_r5_20260401_120636`
- budget: `100` (init `20`)
- policy: `bo`
- seed: `20270051`
- xi: `0.1`
- num_reads: `1000`
- sampler_config: `sampler_config_qpu_reads1000_v113_default.json`
- embed_reuse: `False`

## Best observed query

- step: `49`
- (kdist, kmono): `(7.500, 19.000)`
- objective: `0.837000`
- q_3nq9 / q_4jsz: `0.837000` / `0.002000`
- v_3nq9 / v_4jsz: `0.929000` / `0.995000`

## Notes

- This is an online sequential BO run: each point is evaluated before selecting the next point.
- `quality_min` is the manuscript-aligned quality-centric robust objective.
