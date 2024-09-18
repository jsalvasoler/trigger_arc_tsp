from __future__ import annotations

import os

import gurobipy as gp

from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR, MODELS_DIR, cleanup_instance_name


class GurobiModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.model = gp.Model("TriggerArcTSP")
        self.formulated = False

        self.x, self.y, self.u, self.z = None, None, None, None
        self.u_var_indices = [i for i in self.instance.nodes if i != 0]
        self.z_var_indices = self.instance.z_var_indices

    def get_model_from_model_file(self) -> None | gp.Model:
        model_path = os.path.join(MODELS_DIR, self.instance.model_name)
        if not os.path.exists(model_path):
            return None
        return gp.read(model_path)

    def define_variables_from_model(self) -> None:
        print("Defining variables from model")

        # Retrieve all variables once and store them in a list
        variables = self.model.getVars()

        # Assign variables to x, u, y, and z using slices of the list
        num_x = len(self.instance.edges)
        num_u = len(self.u_var_indices)
        num_y = len(self.instance.relations)
        num_z = len(self.z_var_indices)

        self.x = {key: variables[idx] for idx, key in enumerate(self.instance.edges)}
        self.u = {key: variables[num_x + idx] for idx, key in enumerate(self.u_var_indices)}
        self.y = {key: variables[num_x + num_u + idx] for idx, key in enumerate(self.instance.relations)}
        self.z = {key: variables[num_x + num_u + num_y + idx] for idx, key in enumerate(self.z_var_indices)}

        # Ensure the total number of variables matches
        assert num_x + num_u + num_y + num_z == len(variables)

    def add_extra_const_variables(self) -> None:
        self.u[0] = 0
        for i in self.instance.delta_out[0]:
            for b in self.instance.edges:
                if b == (0, i) or (0, i, *b) not in self.z:
                    continue
                self.z[0, i, *b] = 1

    def formulate(
        self,
        *,
        relax_obj_modeling: bool = False,
        read_model: bool = False,
        vars_: None | list[dict, dict, dict, dict] = None,
    ) -> None:
        self.formulated = True

        if vars_ is not None:
            assert not read_model
            assert not relax_obj_modeling
            assert len(vars_) == 4

        if read_model:
            assert not relax_obj_modeling
            read_model = self.get_model_from_model_file()
            if read_model is not None:
                self.model = read_model
                self.define_variables_from_model()
                self.add_extra_const_variables()
                return

        if vars_:
            # The user has already loaded the variables into the model, we just need to register them
            assert len(self.model.getVars()) > 0
            self.x, self.y, self.u, self.z = vars_
        else:
            if relax_obj_modeling:
                self.add_variables_relax_obj()
            else:
                self.add_variables()
            self.add_extra_const_variables()

        self.add_constraints()

        if not relax_obj_modeling:
            self.model.write(os.path.join(MODELS_DIR, self.instance.model_name))

    def add_variables(self) -> None:
        self.x = self.model.addVars(self.instance.edges, vtype=gp.GRB.BINARY, name="x")
        self.u = self.model.addVars(self.u_var_indices, vtype=gp.GRB.CONTINUOUS, name="u", lb=0, ub=self.instance.N - 1)
        self.y = self.model.addVars(self.instance.relations, vtype=gp.GRB.BINARY, name="y")
        self.z = self.model.addVars(self.z_var_indices, vtype=gp.GRB.BINARY, name="z")

    def add_variables_relax_obj(self) -> None:
        self.x = self.model.addVars(self.instance.edges, vtype=gp.GRB.BINARY, name="x")
        self.u = self.model.addVars(self.u_var_indices, vtype=gp.GRB.CONTINUOUS, name="u", lb=0, ub=self.instance.N - 1)
        self.y = self.model.addVars(self.instance.relations, vtype=gp.GRB.CONTINUOUS, name="y", lb=0, ub=1)
        self.z = self.model.addVars(self.z_var_indices, vtype=gp.GRB.CONTINUOUS, name="z", lb=0, ub=1)

    def add_constraints(self) -> None:
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
            (self.y[*b, *a] <= self.z[*b, *a] for a in self.instance.R_a for b in self.instance.R_a[a]),
            name="trigger_before_arc",
        )

        # If we have u[b] < u[a] in the tour, and x[a] == x[b] == 1, then sum(y[c, a] for c in R_a) == 1
        # (at least one relation is active)
        self.model.addConstrs(
            (
                1 - self.z[*a, *b]
                <= gp.quicksum(self.y[*b, *a] for b in self.instance.R_a[a]) + (1 - self.x[a]) + (1 - self.x[b])
                for a in self.instance.R_a
                for b in self.instance.R_a[a]
            ),
            name="force_relation_active",
        )

        # z[a_1, a_2] == 1 implies u[a_1] <= u[a_2]
        # u[a_2] + 1 <= u[a_1] implies z[a_1, a_2] == 0
        self.model.addConstrs(
            (
                self.u[a10] <= self.u[a20] + (self.instance.N - 1) * (1 - self.z[a10, a11, a20, a21])
                for a10, a11, a20, a21 in self.z
            ),
            name="model_z_variables_1",
        )
        self.model.addConstrs(
            (
                self.z[a10, a11, a20, a21] == 1 - self.z[a20, a21, a10, a11]
                for a10, a11, a20, a21 in self.z
                if a10 != a20
            ),
            name="model_z_variables_2",
        )
        self.model.addConstrs(
            (self.z[a10, a11, a20, a21] == self.z[a20, a21, a10, a11] for a10, a11, a20, a21 in self.z if a10 == a20),
            name="model_z_variables_3",
        )

        # If u[b] < u[c] < u[a] in the tour, then y[b, a] <= y[c, a]
        self.model.addConstrs(
            (
                self.y[*b, *a]
                <= self.z[*c, *b]  # u[b] + 1 < u[c] implies z[c, b] == 0
                + self.z[*a, *c]  # u[c] + 1 < u[a] implies z[c, a] == 0
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

        # Set the objective function
        obj = gp.quicksum(
            self.x[a] * self.instance.edges[a]
            + gp.quicksum(self.y[*b, *a] * self.instance.relations[*b, *a] for b in self.instance.R_a[a])
            for a in self.instance.edges
        )
        self.model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    def provide_mip_start(self, vars_: list[dict]) -> None:
        print("Providing MIP start")
        assert len(vars_) == 4
        for var_name, var in zip(["x", "y", "u", "z"], vars_):
            gb_var = getattr(self, var_name)
            for key, val in var.items():
                if type(gb_var[key]) is gp.Var:
                    gb_var[key].Start = val

    def solve_model_with_parameters(
        self, time_limit_sec: int = 60, heuristic_effort: float = 0.05, presolve: int = -1, *, mip_start: bool = False
    ) -> None:
        if not self.formulated:
            raise ValueError("Model is not formulated")

        if mip_start:
            vars_ = self.instance.get_mip_start()
            self.provide_mip_start(vars_)

        # Set parameters and optimize
        if time_limit_sec > 0:
            self.model.setParam(gp.GRB.Param.TimeLimit, time_limit_sec)
        self.model.setParam(gp.GRB.Param.Heuristics, heuristic_effort)
        # self.model.setParam(gp.GRB.Param.MIPFocus, 1)
        assert presolve in [-1, 0, 1, 2]
        self.model.setParam(gp.GRB.Param.Presolve, presolve)
        self.model.optimize()

        status = self.model.status
        if status == gp.GRB.Status.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")
            raise ValueError("Model is infeasible")

    def get_original_solution(self) -> list[list, float]:
        u_vals = {i: self.u[i].X if type(self.u[i]) is gp.Var else self.u[i] for i in self.u}
        x_vals = {a: self.x[a].X if type(self.x[a]) is gp.Var else self.x[a] for a in self.x}
        y_vals = {a: self.y[a].X if type(self.y[a]) is gp.Var else self.y[a] for a in self.y}

        tour = sorted([i for i in self.instance.nodes if i != 0], key=lambda i: u_vals[i])
        tour = [0, *tour]

        cost = 0
        cost += sum(self.instance.edges[a] for a in self.instance.edges if x_vals[a] > 0.5)
        cost += sum(self.instance.relations[a] for a in self.instance.relations if y_vals[a] > 0.5)

        return tour, cost

    def get_x(self) -> dict:
        return self.x

    def get_y(self) -> dict:
        return self.y

    def get_u(self) -> dict:
        return self.u

    def get_z(self) -> dict:
        return self.z

    def get_model(self) -> gp.Model:
        return self.model


def gurobi_main(
    instance_name: str,
    time_limit_sec: int = 60,
    heuristic_effort: float = 0.05,
    presolve: int = -1,
    *,
    mip_start: bool = False,
    relax_obj_modeling: bool = False,
    read_model: bool = False,
) -> None:
    instance_name = cleanup_instance_name(instance_name)

    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, instance_name))

    model = GurobiModel(instance)
    model.formulate(relax_obj_modeling=relax_obj_modeling, read_model=read_model)
    model.solve_model_with_parameters(
        time_limit_sec=time_limit_sec, heuristic_effort=heuristic_effort, mip_start=mip_start, presolve=presolve
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
        read_model=False,
    )
