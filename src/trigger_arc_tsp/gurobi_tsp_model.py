from __future__ import annotations

from typing import TYPE_CHECKING

import gurobipy as gp

if TYPE_CHECKING:
    from trigger_arc_tsp.instance import Instance


class GurobiTSPModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.model = None

        self.x, self.u = None, None

    def formulate(self) -> None:
        self.model = gp.Model("TriggerArcTSP")

        self.x = self.model.addVars(self.instance.edges, vtype=gp.GRB.BINARY, name="x")
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

        # Set the objective function
        self.model.setObjective(
            gp.quicksum(self.x[a] * self.instance.edges[a] for a in self.instance.edges),
            sense=gp.GRB.MINIMIZE,
        )

    def solve_to_feasible_solution(self) -> None:
        self.check_model_is_formulated()

        # Set parameters and optimize
        self.model.setParam(gp.GRB.Param.SolutionLimit, 1)
        self.model.optimize()

        self.check_model_status()

    def solve_to_optimality(self) -> None:
        self.check_model_is_formulated()

        # Set parameters and optimize
        self.model.optimize()

        self.check_model_status()

    def check_model_is_formulated(self) -> None:
        if self.model is None:
            raise ValueError("Model is not formulated")

    def check_model_status(self) -> None:
        status = self.model.status
        if status == gp.GRB.Status.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")
            raise ValueError("Model is infeasible")

    def get_tour(self) -> list[int]:
        tour = sorted([i for i in self.instance.nodes if i != 0], key=lambda i: self.u[i].X)
        return [0, *tour]

    def get_x(self) -> dict:
        return self.x

    def get_u(self) -> dict:
        return self.u

    def get_model(self) -> gp.Model:
        return self.model
