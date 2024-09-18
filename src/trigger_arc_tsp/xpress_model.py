from __future__ import annotations

import os

import xpress as xp

from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.solver_model import SolverModel
from trigger_arc_tsp.utils import INSTANCES_DIR, cleanup_instance_name


class XpressModel(SolverModel):
    def __init__(self, instance: Instance) -> None:
        super().__init__(instance)
        self.model = xp.problem(name="TriggerArcTSP")

    def read_model(self, model_path: str | os.PathLike) -> xp.problem:
        self.model.read(model_path)
        return self.model

    def get_vars(self) -> list:
        return self.model.getVariable()

    def write_mps(self, model_path: str | os.PathLike) -> None:
        """Do not write the model to a file"""

    def add_variables(self) -> None:
        self.x = self.model.addVariables(list(self.instance.edges.keys()), vartype=xp.binary, name="x")
        self.u = self.model.addVariables(
            self.u_var_indices, vartype=xp.continuous, name="u", lb=0, ub=self.instance.N - 1
        )
        self.y = self.model.addVariables(list(self.instance.relations) or [1], vartype=xp.binary, name="y")
        self.z = self.model.addVariables(self.z_var_indices or [1], vartype=xp.binary, name="z")

    def add_variables_relax_obj(self) -> None:
        self.x = self.model.addVars(list(self.instance.edges), vartype=xp.binary, name="x")
        self.u = self.model.addVars(self.u_var_indices, vartype=xp.continuous, name="u", lb=0, ub=self.instance.N - 1)
        self.y = self.model.addVars(list(self.instance.relations) or [1], vartype=xp.continuous, name="y", lb=0, ub=1)
        self.z = self.model.addVars(self.z_var_indices or [1], vartype=xp.continuous, name="z", lb=0, ub=1)

    def add_constraints(self) -> None:
        raise NotImplementedError("XPRESS cannot model. Use read_model=True instead.")

    def add_objective(self) -> None:
        raise NotImplementedError("XPRESS cannot model. Use read_model=True instead.")

    def provide_mip_start(self, vars_: list[dict]) -> None:
        raise NotImplementedError("XPRESS cannot pass MIP starts.")

    def solve_model_with_parameters(
        self, time_limit_sec: int = 60, heuristic_effort: float = 1, presolve: int = -1, *, mip_start: bool = False
    ) -> None:
        if not self.formulated:
            raise ValueError("Model is not formulated")

        if mip_start:
            vars_ = self.instance.get_mip_start()
            self.provide_mip_start(vars_)

        # Set parameters and optimize
        if time_limit_sec > 0:
            self.model.controls.timelimit = time_limit_sec
        self.model.controls.heursearcheffort = heuristic_effort * 10
        assert presolve in [-1, 0, 1, 2]
        self.model.controls.presolve = presolve
        self.model.optimize()

        status = self.model.attributes.solstatus
        if status == xp.SolStatus.FEASIBLE:
            self.model.iisall()
            raise ValueError("Model is infeasible")

    def get_original_solution(self) -> list[list, float]:
        u_vals = {
            i: self.model.getSolution(self.u[i]) if type(self.u[i]) is xp.var else self.u[i] for i in self.u_var_indices
        }
        x_vals = {a: self.model.getSolution(self.x[a]) if type(self.x[a]) is xp.var else self.x[a] for a in self.x}
        y_vals = {a: self.model.getSolution(self.y[a]) if type(self.y[a]) is xp.var else self.y[a] for a in self.y}

        tour = sorted([i for i in self.instance.nodes if i != 0], key=lambda i: u_vals[i])
        tour = [0, *tour]

        cost = 0
        cost += sum(self.instance.edges[a] for a in self.instance.edges if x_vals[a] > 0.5)
        cost += sum(self.instance.relations[a] for a in self.instance.relations if y_vals[a] > 0.5)

        return tour, cost


def xpress_main(
    instance_name: str,
    time_limit_sec: int = 60,
    heuristic_effort: float = 1,
    presolve: int = -1,
    *,
    mip_start: bool = False,
    relax_obj_modeling: bool = False,
    read_model: bool = False,
) -> None:
    assert not relax_obj_modeling, "XPRESS cannot model. Use read_model=True instead."
    assert read_model, "XPRESS cannot model. read_model=True is required."

    instance_name = cleanup_instance_name(instance_name)

    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, instance_name))

    model = XpressModel(instance)
    model.formulate(relax_obj_modeling=relax_obj_modeling, read_model=read_model)
    model.solve_model_with_parameters(
        time_limit_sec=time_limit_sec, heuristic_effort=heuristic_effort, mip_start=mip_start, presolve=presolve
    )
    tour, cost = model.get_original_solution()

    instance.save_solution(tour, cost)

    print("-" * 60)
    print(f"Instance: {instance_name}")
    print(f"Tour: {tour}")
    print(f"Cost: {cost}")
    print("-" * 60)


if __name__ == "__main__":
    xpress_main(
        instance_name="instances_release_1/grf5.txt",
        time_limit_sec=600,
        mip_start=True,
        heuristic_effort=0.8,
        relax_obj_modeling=True,
        read_model=False,
    )
