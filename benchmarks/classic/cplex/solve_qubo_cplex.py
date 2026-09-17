#!/usr/bin/env python3
"""Solve QDock QUBO (.npy dict) with CPLEX via DOcplex MIQP.

Usage:
  python solve_qubo_cplex.py --qubo /path/to/3nq9_ligand.npy --time-limit 60 --threads 8 \
    --mip-gap 0.01 --output result.json --write-lp model.lp

Optional:
  --max-vars 120 --random-subset --seed 1  # reduce to small QUBO for optimality check
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from docplex.mp.model import Model


def load_qubo(path: Path) -> Dict[Tuple[str, str], float]:
    obj = np.load(path, allow_pickle=True)
    if obj.dtype != object:
        raise ValueError(f"Unexpected QUBO array dtype: {obj.dtype}")
    qubo = obj.item()
    if not isinstance(qubo, dict):
        raise ValueError("QUBO file does not contain a dict")
    return qubo


def build_var_list(qubo: Dict[Tuple[str, str], float]) -> list[str]:
    vars_set = set()
    for u, v in qubo.keys():
        vars_set.add(u)
        vars_set.add(v)
    return sorted(vars_set)


def filter_qubo(
    qubo: Dict[Tuple[str, str], float],
    vars_keep: set[str],
) -> Dict[Tuple[str, str], float]:
    out = {}
    for (u, v), c in qubo.items():
        if u in vars_keep and v in vars_keep:
            out[(u, v)] = c
    return out


def qubo_energy(qubo: Dict[Tuple[str, str], float], sol: Dict[str, int]) -> float:
    e = 0.0
    for (u, v), c in qubo.items():
        e += c * sol.get(u, 0) * sol.get(v, 0)
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubo", required=True, type=Path)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--mip-gap", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-vars", type=int, default=None)
    ap.add_argument("--random-subset", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--write-lp", type=Path, default=None)
    ap.add_argument("--dump-vars", type=Path, default=None)
    args = ap.parse_args()

    qubo = load_qubo(args.qubo)
    vars_all = build_var_list(qubo)

    # Optional size reduction
    vars_use = vars_all
    if args.max_vars and len(vars_all) > args.max_vars:
        rng = random.Random(args.seed)
        if args.random_subset:
            vars_use = sorted(rng.sample(vars_all, args.max_vars))
        else:
            vars_use = vars_all[: args.max_vars]
        qubo = filter_qubo(qubo, set(vars_use))

    if args.dump_vars:
        args.dump_vars.write_text("\n".join(vars_use))

    idx = {v: i for i, v in enumerate(vars_use)}

    mdl = Model(name="qdock_qubo")
    mdl.parameters.timelimit = args.time_limit
    mdl.parameters.mip.tolerances.mipgap = args.mip_gap
    mdl.parameters.threads = args.threads
    mdl.parameters.randomseed = args.seed

    x = mdl.binary_var_list(len(vars_use), name="x")

    # Build quadratic objective
    expr = 0
    for (u, v), c in qubo.items():
        i = idx[u]
        j = idx[v]
        if i == j:
            expr += c * x[i]
        else:
            expr += c * x[i] * x[j]
    mdl.minimize(expr)

    if args.write_lp:
        mdl.export_as_lp(str(args.write_lp))

    sol = mdl.solve(log_output=True)

    status = str(mdl.get_solve_status())
    runtime = mdl.solve_details.time if mdl.solve_details else None
    obj_val = sol.objective_value if sol is not None else None

    sol_map = {}
    if sol is not None:
        for i, v in enumerate(vars_use):
            sol_map[v] = int(round(sol[x[i]]))

    energy = qubo_energy(qubo, sol_map) if sol is not None else None

    result = {
        "qubo_path": str(args.qubo),
        "n_vars": len(vars_use),
        "n_terms": len(qubo),
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
        "threads": args.threads,
        "status": status,
        "runtime_sec": runtime,
        "obj_val": obj_val,
        "energy_from_qubo": energy,
        "solution_ones": [v for v, val in sol_map.items() if val == 1],
    }

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
