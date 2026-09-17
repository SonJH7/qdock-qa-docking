# Software environment records

The two captured environments use Python 3.9.23 and cover QPU sampling and
classical evaluation.

- `pip_freeze_qdock-qpu.txt`: QPU sampling, minor embedding, and D-Wave calls.
- `pip_freeze_qdock.txt`: sample decoding, pose reconstruction, and RMSD
  evaluation.

## Key versions used for the QPU runs (env `qdock-qpu`, Python 3.9.23)
dwave-ocean-sdk 9.0.0 | dwave-system 1.33.0 | minorminer 0.2.19 | dimod 0.12.21 | dwave-neal 0.6.0 |
dwave-hybrid 0.6.14 | dwave-cloud-client 0.14.0 | dwave-preprocessing 0.6.10 | dwave-samplers 1.6.0 |
dwave-optimization 0.6.4 (NL hybrid) | numpy 1.23.5 | scipy 1.13.1 | pyqubo 1.5.0.
(Gurobi 12.0.3 / CPLEX 22.1.2 are separate solvers, stated in the manuscript.)

The sampling environment used NumPy 1.23.5. The evaluation environment used
NumPy 1.22.4.
