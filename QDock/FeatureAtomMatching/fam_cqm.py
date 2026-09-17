from dimod import ConstrainedQuadraticModel, Binary
from itertools import combinations

def _ligand_element(atom_type):
    if atom_type == "A":
        return "C"
    return atom_type


def build_fam_cqm(ligand, feature_atoms, box_dmatrix, w_dict, edge_cutoff,
                  K_dist=None, K_mono=None, max_constraints=None):
    cqm = ConstrainedQuadraticModel()
    vertices = []
    objective = 0

    for i in range(ligand.n):
        t_lig = _ligand_element(ligand.autodock_atom_types[i])
        if t_lig not in w_dict:
            t_lig = t_lig[0]
        for j, fs in enumerate(feature_atoms):
            t_fs = fs.getElement()
            weight = abs(w_dict[t_lig] - w_dict[t_fs]) - 0.5
            x = Binary("%d_%d" % (i, j))
            vertices.append((x, i, j, weight))
            objective += weight * x

    cqm.set_objective(objective)

    by_atom = {
        i: [x for x, ii, _, _ in vertices if ii == i]
        for i in range(ligand.n)
    }
    by_feature = {
        j: [x for x, _, jj, _ in vertices if jj == j]
        for j in range(len(feature_atoms))
    }

    # E_mono
    if K_mono is None:
        # Hard CQM: use the same feasible set as E_mono = 0.
        for i, xs in by_atom.items():
            cqm.add_constraint(
                sum(xs) == 1,
                label="atom_exact_%d" % i,
            )

        for j, xs in by_feature.items():
            if len(xs) > 1:
                cqm.add_constraint(
                    sum(xs) <= 1,
                    label="feature_unique_%d" % j,
                )
    else:
        # Soft CQM: expand K_mono * E_mono exactly.
        km = float(K_mono)

        for xs in by_atom.values():
            objective += km
            for x in xs:
                objective -= km * x
            for x1, x2 in combinations(xs, 2):
                objective += 2.0 * km * x1 * x2

        for xs in by_feature.values():
            for x1, x2 in combinations(xs, 2):
                objective += km * x1 * x2

    # E_dist: inspect only distinct ligand-atom pairs, as in the paper.
    added = 0
    limit_hit = False

    for p in range(len(vertices)):
        x1, i1, j1, _ = vertices[p]

        for q in range(p + 1, len(vertices)):
            x2, i2, j2, _ = vertices[q]

            if i1 == i2:
                continue

            dd = abs(
                ligand.d_matrix[i1, i2]
                - box_dmatrix[j1, j2]
            )
            if dd <= edge_cutoff:
                continue

            if max_constraints is not None and added >= max_constraints:
                raise RuntimeError(
                    "Distance constraint limit reached; refusing a partial FAM model."
                )

            if K_dist is None:
                cqm.add_constraint(
                    x1 + x2 <= 1,
                    label="dist_%d_%d_%d_%d" % (i1, j1, i2, j2),
                )
            else:
                objective += float(K_dist) * x1 * x2

            added += 1

    cqm.set_objective(objective)
    return cqm, vertices, added, limit_hit


def build_fam_cqm_spec(ligand, feature_atoms, box_dmatrix, w_dict, edge_cutoff,
                       K_dist=None, K_mono=None, max_constraints=None):
    vertices = []
    objective_linear = []
    objective_quadratic = []
    constraints = []

    km = None if K_mono is None else float(K_mono)
    kd = None if K_dist is None else float(K_dist)

    # Constant term from K_mono * (1 - sum_j x_ij)^2.
    objective_offset = (
        float(ligand.n) * km
        if km is not None
        else 0.0
    )

    # Create all x_ij variables and linear objective terms.
    for i in range(ligand.n):
        t_lig = _ligand_element(ligand.autodock_atom_types[i])
        if t_lig not in w_dict:
            t_lig = t_lig[0]

        for j, fs in enumerate(feature_atoms):
            t_fs = fs.getElement()
            weight = abs(w_dict[t_lig] - w_dict[t_fs]) - 0.5
            name = "%d_%d" % (i, j)

            linear_weight = float(weight)
            if km is not None:
                linear_weight -= km

            vertices.append((name, i, j, weight))
            objective_linear.append((name, linear_weight))

    by_atom = {
        i: [name for name, ii, _, _ in vertices if ii == i]
        for i in range(ligand.n)
    }
    by_feature = {
        j: [name for name, _, jj, _ in vertices if jj == j]
        for j in range(len(feature_atoms))
    }

    atom_exact_constraints = 0
    feature_unique_constraints = 0

    if km is None:
        # Hard CQM: each ligand atom must select exactly one feature.
        for i, names in by_atom.items():
            constraints.append({
                "label": "atom_exact_%d" % i,
                "sense": "==",
                "rhs": 1.0,
                "terms": [(name, 1.0) for name in names],
            })
            atom_exact_constraints += 1

        # Hard CQM: each feature can be used by at most one ligand atom.
        for j, names in by_feature.items():
            if len(names) > 1:
                constraints.append({
                    "label": "feature_unique_%d" % j,
                    "sense": "<=",
                    "rhs": 1.0,
                    "terms": [(name, 1.0) for name in names],
                })
                feature_unique_constraints += 1
    else:
        # Soft same-atom terms:
        # 2 * K_mono * sum_i sum_{j<l} x_ij x_il
        for names in by_atom.values():
            for name1, name2 in combinations(names, 2):
                objective_quadratic.append(
                    (name1, name2, 2.0 * km)
                )

        # Soft shared-feature terms:
        # K_mono * sum_j sum_{i<k} x_ij x_kj
        for names in by_feature.values():
            for name1, name2 in combinations(names, 2):
                objective_quadratic.append(
                    (name1, name2, km)
                )

    # Distance incompatibilities are defined only for different ligand atoms.
    dist_constraints = 0
    limit_hit = False

    for p in range(len(vertices)):
        name1, i1, j1, _ = vertices[p]

        for q in range(p + 1, len(vertices)):
            name2, i2, j2, _ = vertices[q]

            if i1 == i2:
                continue

            dd = abs(
                ligand.d_matrix[i1, i2]
                - box_dmatrix[j1, j2]
            )
            if dd <= edge_cutoff:
                continue

            if (
                max_constraints is not None
                and dist_constraints >= max_constraints
            ):
                raise RuntimeError(
                    "Distance constraint limit reached; "
                    "refusing a partial FAM model."
                )

            if kd is None:
                constraints.append({
                    "label": "dist_%d_%d_%d_%d"
                             % (i1, j1, i2, j2),
                    "sense": "<=",
                    "rhs": 1.0,
                    "terms": [
                        (name1, 1.0),
                        (name2, 1.0),
                    ],
                })
            else:
                objective_quadratic.append(
                    (name1, name2, kd)
                )

            dist_constraints += 1

    return {
        "variables": [
            name for name, _, _, _ in vertices
        ],
        "objective_offset": objective_offset,
        "objective_linear": objective_linear,
        "objective_quadratic": objective_quadratic,
        "constraints": constraints,
        "counts": {
            "num_vars": len(vertices),
            "mono_constraints": (
                atom_exact_constraints
                + feature_unique_constraints
            ),
            "atom_exact_constraints": atom_exact_constraints,
            "feature_unique_constraints": feature_unique_constraints,
            "dist_constraints": dist_constraints,
            "constraint_limit_hit": limit_hit,
        },
        "params": {
            "edge_cutoff": edge_cutoff,
            "K_dist": K_dist,
            "K_mono": K_mono,
            "max_constraints": max_constraints,
        },
    }
