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

        self.x, self.y, self.u, self.z = None, None, None, None
        self.z_indices = [(a, b) for a in self.instance.edges for b in self.instance.edges if a != b]

    def formulate(self, *, relax_obj_modeling: bool = False) -> None:
        self.model = gp.Model("TriggerArcTSP")

        self.x = self.model.addVars(self.instance.edges, vtype=gp.GRB.BINARY, name="x")
        self.u = self.model.addVars(
            self.instance.nodes, vtype=gp.GRB.CONTINUOUS, name="u", lb=0, ub=self.instance.N - 1
        )
        if not relax_obj_modeling:
            self.y = self.model.addVars(self.instance.relations, vtype=gp.GRB.BINARY, name="y")
            self.z = self.model.addVars(self.z_indices, vtype=gp.GRB.BINARY, name="z")
        else:
            self.y = self.model.addVars(self.instance.relations, vtype=gp.GRB.CONTINUOUS, name="y", lb=0, ub=1)
            self.z = self.model.addVars(self.z_indices, vtype=gp.GRB.CONTINUOUS, name="z", lb=0, ub=1)

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

        # If we have u[b] < u[a] in the tour, and x[a] == x[b] == 1, then sum(y[c, a] for c in R_a) == 1
        # (at least one relation is active)
        self.model.addConstrs(
            (1 - self.z[a, b])
            <= gp.quicksum(self.y[*b, *a] for b in self.instance.R_a[a]) + (1 - self.x[a]) + (1 - self.x[b])
            for a in self.instance.R_a
            for b in self.instance.R_a[a]
        )

        # z[a_1, a_2] == 1 implies u[a_1] <= u[a_2]
        # u[a_2] + 1 <= u[a_1] implies z[a_1, a_2] == 0
        self.model.addConstrs(
            (
                self.u[a_1[0]] <= self.u[a_2[0]] + (self.instance.N - 1) * (1 - self.z[a_1, a_2])
                for a_1, a_2 in self.z_indices
            ),
            name="model_z_variables",
        )

        # If u[b] < u[c] < u[a] in the tour, then y[b, a] <= y[c, a]
        self.model.addConstrs(
            (
                self.y[*b, *a]
                <= self.y[*c, *a]
                + self.z[c, b]  # u[b] + 1 < u[c] implies z[c, b] == 0
                + self.z[a, c]  # u[c] + 1 < u[a] implies z[c, a] == 0
                + (1 - self.x[c])
                + (1 - self.x[b])
                + (1 - self.x[a])
                for a in self.instance.R_a
                for b in self.instance.R_a[a]
                for c in self.instance.R_a[a]
                if b != c
            ),
            name="only_last_relation_triggers",
        )

        # Set u_0 = 0
        self.model.addConstr(self.u[0] == 0, name="depot_starts_sequence_at_zero")

        # Set the objective function
        obj = gp.quicksum(
            self.x[a] * self.instance.edges[a]
            + gp.quicksum(self.y[*b, *a] * self.instance.relations[*b, *a] for b in self.instance.R_a[a])
            for a in self.instance.edges
        )
        self.model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    def provide_mip_start(self, vars_: list[dict]) -> None:
        assert len(vars_) == 4
        for var_name, var in zip(["x", "y", "u", "z"], vars_):
            gb_var = getattr(self, var_name)
            for key, val in var.items():
                gb_var[key].Start = val

    def solve_model_with_parameters(
        self, time_limit_sec: int = 60, heuristic_effort: float = 0.05, *, mip_start: bool = False
    ) -> None:
        if self.model is None:
            raise ValueError("Model is not formulated")

        if mip_start:
            vars_ = self.instance.get_mip_start()
            self.provide_mip_start(vars_)

        # Set parameters and optimize
        self.model.setParam(gp.GRB.Param.TimeLimit, time_limit_sec)
        self.model.setParam(gp.GRB.Param.Heuristics, heuristic_effort)
        self.model.optimize()

        status = self.model.status
        if status == gp.GRB.Status.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")
            raise ValueError("Model is infeasible")

    def get_original_solution(self, *, keep_offset: bool = False) -> list[list, float]:
        tour = sorted([i for i in self.instance.nodes if i != 0], key=lambda i: self.u[i].X)
        tour = [0, *tour]

        cost = 0
        cost += sum(self.instance.edges[a] for a in self.instance.edges if self.x[a].X > 0.5)
        offset = 0 if keep_offset else self.instance.offset
        cost += sum(self.instance.relations[a] - offset for a in self.instance.relations if self.y[a].X > 0.5)

        return tour, cost

    def get_x(self) -> dict:
        return self.x

    def get_y(self) -> dict:
        return self.y

    def get_u(self) -> dict:
        return self.u

    def get_model(self) -> gp.Model:
        return self.model


def gurobi_main(
    instance_name: str,
    time_limit_sec: int = 60,
    heuristic_effort: float = 0.05,
    *,
    mip_start: bool = False,
    relax_obj_modeling: bool = False,
) -> None:
    instance_name = cleanup_instance_name(instance_name)

    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, instance_name))

    model = GurobiModel(instance)
    model.formulate(relax_obj_modeling=relax_obj_modeling)
    model.solve_model_with_parameters(
        time_limit_sec=time_limit_sec, heuristic_effort=heuristic_effort, mip_start=mip_start
    )
    tour, cost = model.get_original_solution()

    instance.save_solution(tour, cost)

    # tour = [0,8,6,4,18,19,5,14,3,1,15,13,10,7,9,2,16,12,11,17]
    # cost = instance.compute_objective(tour)

    print("-" * 60)
    print(f"Instance: {instance_name}")
    print(f"Tour: {tour}")
    print(f"Cost: {cost}")
    print("-" * 60)


if __name__ == "__main__":
    gurobi_main(
        instance_name="instances_release_1/grf5.txt",
        time_limit_sec=600,
        mip_start=True,
        heuristic_effort=0.8,
        relax_obj_modeling=True,
    )
