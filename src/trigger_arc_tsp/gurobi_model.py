import os

import gurobipy as gp

from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR


def main() -> None:
    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples", "example_1.txt"))
    print(instance)

    model = gp.Model("TriggerArcTSP")

    x = model.addVars(instance.edges, vtype=gp.GRB.BINARY, name="x")
    y = model.addVars(instance.relations, vtype=gp.GRB.BINARY, name="y")
    u = model.addVars(instance.nodes, vtype=gp.GRB.CONTINUOUS, name="u")

    # Flow conservation constraints
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in instance.delta_out[i]) == 1 for i in instance.nodes), name="flow_conservation_out"
    )
    model.addConstrs(
        (gp.quicksum(x[j, i] for j in instance.delta_in[i]) == 1 for i in instance.nodes), name="flow_conservation_in"
    )

    # Subtour elimination constraints
    model.addConstrs(
        (u[i] - u[j] + instance.N * x[i, j] <= instance.N - 1 for i, j in instance.edges if i != 0 and j != 0),
        name="subtour_elimination",
    )

    # At most one relation can be active in R_a
    model.addConstrs(
        (gp.quicksum(y[*b, *a] for b in instance.R_a[a]) <= x[a] for a in instance.R_a), name="max_one_relation"
    )

    # Relation r=(b,a) \in R_a is inactive if a or b are inactive
    model.addConstrs(
        (y[*b, *a] <= x[a] for a in instance.R_a for b in instance.R_a[a]), name="relation_inactive_if_target_inactive"
    )
    model.addConstrs(
        (y[*b, *a] <= x[b] for a in instance.R_a for b in instance.R_a[a]), name="relation_inactive_if_target_inactive"
    )

    # Relation r=(b,a) \in R_a is inactive if b after a in the tour
    model.addConstrs(
        (u[b[0]] + 1 <= u[a[0]] + instance.N * (1 - y[*b, *a]) for a in instance.R_a for b in instance.R_a[a]),
        name="trigger_before_arc",
    )

    # For different relations r=(b,a) and r'=(c,a) in R_a, if c comes after b
    # in the tour, then y[b, a] <= y[c, a] (1 <= 1 or 0 <= 0 or 0 <= 1)
    z_1, z_2 = {}, {}
    for a in instance.R_a:
        for b in instance.R_a[a]:
            for c in instance.R_a[a]:
                if b == c:
                    continue
                # add two binary variables z_1 and z_2 to model the implication
                z_1[a, b, c] = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f"z_1_{b}_{c}_{a}",
                )
                z_2[a, b, c] = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f"z_2_{b}_{c}_{a}",
                )
                # If in the tour we have b -> c -> a, then y[b, a] = 0
                model.addConstr(u[b[0]] + 1 <= u[c[0]] + instance.N * (z_1[a, b, c]))
                model.addConstr(u[c[0]] <= u[b[0]] + instance.N * (1 - z_1[a, b, c]))
                model.addConstr(u[c[0]] + 1 <= u[a[0]] + instance.N * (z_2[a, b, c]))
                model.addConstr(u[a[0]] <= u[c[0]] + instance.N * (1 - z_2[a, b, c]))
                model.addConstr(y[*b, *a] <= z_1[a, b, c] + z_2[a, b, c])

    # Set the objective function
    model.setObjective(
        gp.quicksum(
            x[a] * instance.edges[a] + gp.quicksum(y[*b, *a] * instance.relations[*b, *a] for b in instance.R_a[a])
            for a in instance.edges
        ),
        sense=gp.GRB.MINIMIZE,
    )

    # # Force the solution with order
    # model.addConstr(u[0] == 0)
    # model.addConstr(u[2] == 1)
    # model.addConstr(u[1] == 2)
    # model.addConstr(u[4] == 3)
    # model.addConstr(u[3] == 4)

    model.optimize()

    status = model.status
    if status == gp.GRB.Status.INFEASIBLE:
        # run infeasibility analysis
        model.computeIIS()
        model.write("model.ilp")
        return

    # print("\n\n")
    # print(" - Solution:")
    baseline_cost = 0
    for i, j in instance.edges:
        if x[i, j].x > 0.5:
            # print(f"Edge ({i}, {j}) is used")
            baseline_cost += instance.edges[i, j]

    # print(f" - Baseline cost: {baseline_cost}")
    for i, j, r, s in instance.relations:
        if y[i, j, r, s].x > 0.5:
            # print(
            #     f" - Relation ({i}, {j}) -> ({r}, {s}) triggers cost "
            #     f"change + {instance.relations[i, j, r, s] - instance.offset}"
            # )
            baseline_cost += instance.relations[i, j, r, s] - instance.offset
        else:
            # print(f" - Relation ({i}, {j}) -> ({r}, {s}) is inactive")
            pass

    # print(f" - Total cost: {baseline_cost}")

    tour = sorted([i for i in instance.nodes if i != 0], key=lambda i: u[i].x)
    tour = [0, *tour, 0]

    instance.test_solution(tour, baseline_cost)
    instance.save_solution(tour, baseline_cost)


if __name__ == "__main__":
    main()
