import gurobipy as gp

from src.trigger_arc_tsp.gurobi_model import GurobiModel
from src.trigger_arc_tsp.instance import Instance


class MILPbasedLNS:
    def __init__(self, instance: Instance, current_tour: list, reinsert_nodes: list) -> None:
        self.instance = instance
        self.current_tour = current_tour
        self.reinsert_nodes = set(reinsert_nodes)

        x, y, u, z = self.instance.get_variables_from_tour(self.current_tour)

        self.model = GurobiModel(self.instance)
        self.gu_model = self.model.get_model()
        self.register_variables_to_model(x, y, u, z)

    def register_variables_to_model(self, x: dict, y: dict, u: dict, z: dict) -> None:
        for i, j in x:
            if i in self.reinsert_nodes or j in self.reinsert_nodes:
                start_val = x[i, j]
                x[i, j] = self.gu_model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}")
                x[i, j].Start = start_val

        for idx in y:
            if any(i in self.reinsert_nodes for i in idx):
                start_val = y[idx]
                y[idx] = self.gu_model.addVar(vtype=gp.GRB.BINARY, name=f"y_{idx}")
                y[idx].Start = start_val

        for i in u:
            start_val = u[i]
            u[i] = self.gu_model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"u_{i}")
            u[i].Start = start_val

        for idx in z:
            if any(i in self.reinsert_nodes for i in idx):
                start_val = z[idx]
                z[idx] = self.gu_model.addVar(vtype=gp.GRB.BINARY, name=f"z_{idx}")
                z[idx].Start = start_val

        self.gu_model.update()
        self.model.formulate(vars_=(x, y, u, z))

        self.gu_model = self.model.get_model()

        print("The LNS MILP model has been formulated.")
        print(f"Number of variables: {self.gu_model.NumVars}")
        print(f"Number of constraints: {self.gu_model.NumConstrs}")
        print(f"Number of non-zero elements: {self.gu_model.NumNZs}")

    def explore(self) -> list:
        # MIP Start is false because it has already been provided
        self.model.solve_model_with_parameters(time_limit_sec=60, heuristic_effort=0.8, mip_start=False)

        return self.model.get_original_solution()
