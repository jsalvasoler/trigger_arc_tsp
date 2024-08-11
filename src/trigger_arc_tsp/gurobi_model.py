from __future__ import annotations

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
        self.model = None

        self.x, self.y, self.u = None, None, None

    def formulate(self) -> None:
        self.model = gp.Model("TriggerArcTSP")

        self.x = self.model.addVars(self.instance.edges, vtype=gp.GRB.BINARY, name="x")
        self.y = self.model.addVars(self.instance.relations, vtype=gp.GRB.BINARY, name="y")
        self.u = self.model.addVars(self.instance.nodes, vtype=gp.GRB.CONTINUOUS, name="u")

        # Flow conservation constraints
        self.model.addConstrs(
            (gp.quicksum(self.x[i, j] for j in self.instance.delta_out[i]) == 1 for i in self.instance.nodes),
            name="flow_conservation_out",
        )
        self.model.addConstrs(
            (gp.quicksum(self.x[j, i] for j in self.instance.delta_in[i]) == 1 for i in self.instance.nodes),
            name="flow_conservation_in",
        )

        # Subtour elimination constraints
        self.model.addConstrs(
            (
                self.u[i] - self.u[j] + self.instance.N * self.x[i, j] <= self.instance.N - 1
                for i, j in self.instance.edges
                if j != 0
            ),
            name="subtour_elimination",
        )

        # At most one relation can be active in R_a
        self.model.addConstrs(
            (gp.quicksum(self.y[*b, *a] for b in self.instance.R_a[a]) <= self.x[a] for a in self.instance.R_a),
            name="max_one_relation",
        )

        # Relation r=(b,a) \in R_a is inactive if a or b are inactive
        self.model.addConstrs(
            (self.y[*b, *a] <= self.x[a] for a in self.instance.R_a for b in self.instance.R_a[a]),
            name="relation_inactive_if_target_inactive",
        )
        self.model.addConstrs(
            (self.y[*b, *a] <= self.x[b] for a in self.instance.R_a for b in self.instance.R_a[a]),
            name="relation_inactive_if_target_inactive",
        )

        # Relation r=(b,a) \in R_a is inactive if b after a in the tour
        self.model.addConstrs(
            (
                self.u[b[0]] + 1 <= self.u[a[0]] + self.instance.N * (1 - self.y[*b, *a])
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
                    # add two binary variables z_1 and z_2 to self.model the implication
                    z_1[a, b, c] = self.model.addVar(
                        vtype=gp.GRB.BINARY,
                        name=f"z_1_{b}_{c}_{a}",
                    )
                    z_2[a, b, c] = self.model.addVar(
                        vtype=gp.GRB.BINARY,
                        name=f"z_2_{b}_{c}_{a}",
                    )
                    # If in the tour we have b -> c -> a, then y[b, a] = 0
                    self.model.addConstr(self.u[b[0]] + 1 <= self.u[c[0]] + self.instance.N * (z_1[a, b, c]))
                    self.model.addConstr(self.u[c[0]] <= self.u[b[0]] + self.instance.N * (1 - z_1[a, b, c]))
                    self.model.addConstr(self.u[c[0]] + 1 <= self.u[a[0]] + self.instance.N * (z_2[a, b, c]))
                    self.model.addConstr(self.u[a[0]] <= self.u[c[0]] + self.instance.N * (1 - z_2[a, b, c]))
                    self.model.addConstr(self.y[*b, *a] <= z_1[a, b, c] + z_2[a, b, c])

        # Set the objective function
        self.model.setObjective(
            gp.quicksum(
                self.x[a] * self.instance.edges[a]
                + gp.quicksum(self.y[*b, *a] * self.instance.relations[*b, *a] for b in self.instance.R_a[a])
                for a in self.instance.edges
            ),
            sense=gp.GRB.MINIMIZE,
        )

    def solve_model_with_parameters(self) -> None:
        if self.model is None:
            raise ValueError("Model is not formulated")

        # set time limit to two minutes
        self.model.setParam(gp.GRB.Param.TimeLimit, 120)
        self.model.optimize()

        status = self.model.status
        if status == gp.GRB.Status.INFEASIBLE:
            # run infeasibility analysis
            self.model.computeIIS()
            self.model.write("model.ilp")
            error_msg = "Model is infeasible"
            raise ValueError(error_msg)

    def get_original_objective(self) -> list[list, float]:
        tour = sorted([i for i in self.instance.nodes if i != 0], key=lambda i: self.u[i].X)
        tour = [0, *tour]

        cost = 0
        for edge in self.instance.edges:
            if self.x[edge].X > 0.5:
                cost += self.instance.edges[edge]

        for relation in self.instance.relations:
            if self.y[relation].X > 0.5:
                cost += self.instance.relations[relation] - self.instance.offset

        return tour, cost

    def get_x(self) -> dict:
        return self.x

    def get_y(self) -> dict:
        return self.y

    def get_u(self) -> dict:
        return self.u

    def get_model(self) -> gp.Model:
        return self.model


def gurobi_main(instance_name: str) -> None:
    instance_name = cleanup_instance_name(instance_name)

    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, instance_name))

    gurobi_model = GurobiModel(instance)
    model = gurobi_model.formulate()
    model = gurobi_model.solve_model_with_parameters(model)
    tour, cost = gurobi_model.get_original_objective(model)

    instance.test_solution(tour, cost)
    instance.save_solution(tour, cost)


if __name__ == "__main__":
    gurobi_main()
