import os

import gurobipy as gp

from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR


def cleanup_instance_name(instance: str) -> str:
    if not instance.endswith(".txt"):
        error_msg = "Instance file must be a .txt file"
        raise ValueError(error_msg)
    if instance.startswith("instances/"):
        instance = instance[10:]

    if instance.count("/") != 1:
        instance = f"instances_release_1/{instance}"

    # check if file exists
    if not os.path.exists(os.path.join(INSTANCES_DIR, instance)):
        error_msg = f"Instance file {instance} not found"
        raise FileNotFoundError(error_msg)

    return instance


class GurobiModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    def formulate(self) -> gp.Model:
        model = gp.Model("TriggerArcTSP")

        x = model.addVars(self.instance.edges, vtype=gp.GRB.BINARY, name="x")
        y = model.addVars(self.instance.relations, vtype=gp.GRB.BINARY, name="y")
        u = model.addVars(self.instance.nodes, vtype=gp.GRB.CONTINUOUS, name="u")

        model._x = x
        model._y = y
        model._u = u

        # Flow conservation constraints
        model.addConstrs(
            (gp.quicksum(x[i, j] for j in self.instance.delta_out[i]) == 1 for i in self.instance.nodes),
            name="flow_conservation_out",
        )
        model.addConstrs(
            (gp.quicksum(x[j, i] for j in self.instance.delta_in[i]) == 1 for i in self.instance.nodes),
            name="flow_conservation_in",
        )

        # Subtour elimination constraints
        model.addConstrs(
            (
                u[i] - u[j] + self.instance.N * x[i, j] <= self.instance.N - 1
                for i, j in self.instance.edges
                if i != 0 and j != 0
            ),
            name="subtour_elimination",
        )

        # At most one relation can be active in R_a
        model.addConstrs(
            (gp.quicksum(y[*b, *a] for b in self.instance.R_a[a]) <= x[a] for a in self.instance.R_a),
            name="max_one_relation",
        )

        # Relation r=(b,a) \in R_a is inactive if a or b are inactive
        model.addConstrs(
            (y[*b, *a] <= x[a] for a in self.instance.R_a for b in self.instance.R_a[a]),
            name="relation_inactive_if_target_inactive",
        )
        model.addConstrs(
            (y[*b, *a] <= x[b] for a in self.instance.R_a for b in self.instance.R_a[a]),
            name="relation_inactive_if_target_inactive",
        )

        # Relation r=(b,a) \in R_a is inactive if b after a in the tour
        model.addConstrs(
            (
                u[b[0]] + 1 <= u[a[0]] + self.instance.N * (1 - y[*b, *a])
                for a in self.instance.R_a
                for b in self.instance.R_a[a]
            ),
            name="trigger_before_arc",
        )

        # For different relations r=(b,a) and r'=(c,a) in R_a, if c comes after b
        # in the tour, then y[b, a] <= y[c, a] (1 <= 1 or 0 <= 0 or 0 <= 1)
        z_1, z_2 = {}, {}
        for a in self.instance.R_a:
            for b in self.instance.R_a[a]:
                for c in self.instance.R_a[a]:
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
                    model.addConstr(u[b[0]] + 1 <= u[c[0]] + self.instance.N * (z_1[a, b, c]))
                    model.addConstr(u[c[0]] <= u[b[0]] + self.instance.N * (1 - z_1[a, b, c]))
                    model.addConstr(u[c[0]] + 1 <= u[a[0]] + self.instance.N * (z_2[a, b, c]))
                    model.addConstr(u[a[0]] <= u[c[0]] + self.instance.N * (1 - z_2[a, b, c]))
                    model.addConstr(y[*b, *a] <= z_1[a, b, c] + z_2[a, b, c])

        # Set the objective function
        model.setObjective(
            gp.quicksum(
                x[a] * self.instance.edges[a]
                + gp.quicksum(y[*b, *a] * self.instance.relations[*b, *a] for b in self.instance.R_a[a])
                for a in self.instance.edges
            ),
            sense=gp.GRB.MINIMIZE,
        )

        return model

    def solve_model_with_parameters(self, model: gp.Model) -> gp.Model:
        # set time limit to two minutes
        model.setParam(gp.GRB.Param.TimeLimit, 120)
        model.optimize()

        status = model.status
        if status == gp.GRB.Status.INFEASIBLE:
            # run infeasibility analysis
            model.computeIIS()
            model.write("model.ilp")
            error_msg = "Model is infeasible"
            raise ValueError(error_msg)

        return model


def gurobi_main(instance_name: str) -> None:
    instance_name = cleanup_instance_name(instance_name)

    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, instance_name))

    gurobi_model = GurobiModel(instance)
    model = gurobi_model.formulate()
    model = gurobi_model.solve_model_with_parameters(model)

    u = model._u
    x = model._x
    y = model._y

    tour = sorted([i for i in instance.nodes if i != 0], key=lambda i: u[i].X)
    tour = [0, *tour]

    cost = 0
    for edge in instance.edges:
        if x[edge].X > 0.5:
            cost += instance.edges[edge]

    for relation in instance.relations:
        if y[relation].X > 0.5:
            cost += instance.relations[relation] + instance.offset

    instance.test_solution(tour, cost)
    instance.save_solution(tour, cost)


if __name__ == "__main__":
    gurobi_main()
