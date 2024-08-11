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

    # Set a random objective
    model.setObjective(gp.quicksum(x) + gp.quicksum(y), sense=gp.GRB.MINIMIZE)

    model.optimize()

    print("Objective value:", model.objVal)


if __name__ == "__main__":
    main()
