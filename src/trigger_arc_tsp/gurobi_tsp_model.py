from __future__ import annotations

from typing import TYPE_CHECKING

import gurobipy as gp

if TYPE_CHECKING:
    from trigger_arc_tsp.instance import Instance


class GurobiTSPModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.model = gp.Model("TriggerArcTSP")
        self.formulated = False

        self.x, self.u = None, None

    def formulate(self) -> None:
        self.formulated = True

        self.x = self.model.addVars(self.instance.edges, vtype=gp.GRB.BINARY, name="x")
        u_index = [i for i in self.instance.nodes if i != 0]
        self.u = self.model.addVars(u_index, vtype=gp.GRB.CONTINUOUS, name="u")
        self.u[0] = 0

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

    def solve_to_optimality(
        self, time_limit_sec: int | None = None, best_bd_stop: float | None = None, *, logs: bool = True
    ) -> None:
        self.check_model_is_formulated()

        if not logs:
            self.model.setParam(gp.GRB.Param.OutputFlag, 0)
        if best_bd_stop:
            self.model.setParam(gp.GRB.Param.BestBdStop, best_bd_stop)
        self.model.setParam(gp.GRB.Param.TimeLimit, time_limit_sec or 60)
        self.model.setParam(gp.GRB.Param.Heuristics, 0.1)

        self.model.optimize()

        self.check_model_status()

    def get_best_tour(self) -> None:
        tour = sorted([i for i in self.instance.nodes if i != 0], key=lambda i: self.u[i].X)
        return [0, *tour]

    def get_best_n_tours(self, n: int) -> list[list[int]]:
        n_solutions = self.model.SolCount
        tours = []
        for i in range(min(n, n_solutions)):
            self.model.setParam(gp.GRB.Param.SolutionNumber, i)
            self.model.update()
            tours.append(self.get_best_tour())
        return tours

    def check_model_is_formulated(self) -> None:
        if not self.formulated:
            raise ValueError("Model is not formulated")

    def check_model_status(self) -> None:
        status = self.model.status
        if status == gp.GRB.Status.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")
            raise ValueError("Model is infeasible")

    def get_x(self) -> dict:
        return self.x

    def get_u(self) -> dict:
        return self.u

    def get_model(self) -> gp.Model:
        return self.model
