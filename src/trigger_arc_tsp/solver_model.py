from __future__ import annotations

import os
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gurobipy as gp
    import xpress as xp

    from trigger_arc_tsp.instance import Instance

from trigger_arc_tsp.utils import MODELS_DIR


class SolverModel:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.formulated = False

        self.x, self.y, self.u, self.z = None, None, None, None
        self.u_var_indices = [i for i in self.instance.nodes if i != 0]
        self.z_var_indices = self.instance.z_var_indices

    @abstractmethod
    def read_model(self, model_path: str | os.PathLike) -> gp.Model | xp.problem:
        pass

    def get_model_from_model_file(self) -> None | gp.Model:
        model_path = os.path.join(MODELS_DIR, self.instance.model_name)
        if not os.path.exists(model_path):
            return None
        return self.read_model(model_path)

    @abstractmethod
    def get_vars(self) -> list:
        pass

    @abstractmethod
    def write_mps(self, model_path: str | os.PathLike) -> None:
        pass

    def define_variables_from_model(self) -> None:
        print("Defining variables from model")

        # Retrieve all variables once and store them in a list
        variables = self.get_vars()

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
            model_from_file = self.get_model_from_model_file()
            if model_from_file is not None:
                self.model = model_from_file
                self.define_variables_from_model()
                self.add_extra_const_variables()
                return

        if vars_:
            # The user has already loaded the variables into the model, we just need to register them
            assert len(self.get_vars()) > 0
            self.x, self.y, self.u, self.z = vars_
        else:
            if relax_obj_modeling:
                self.add_variables_relax_obj()
            else:
                self.add_variables()
            self.add_extra_const_variables()

        self.add_constraints()
        self.add_objective()

        if not relax_obj_modeling:
            self.write_mps(os.path.join(MODELS_DIR, self.instance.model_name))

    @abstractmethod
    def add_variables(self) -> None:
        pass

    @abstractmethod
    def add_variables_relax_obj(self) -> None:
        pass

    @abstractmethod
    def add_constraints(self) -> None:
        pass

    @abstractmethod
    def add_objective(self) -> None:
        pass

    @abstractmethod
    def provide_mip_start(self, vars_: list[dict]) -> None:
        pass

    @abstractmethod
    def solve_model_with_parameters(
        self, time_limit_sec: int = 60, heuristic_effort: float = 0.05, presolve: int = -1, *, mip_start: bool = False
    ) -> None:
        pass

    @abstractmethod
    def get_original_solution(self) -> list[list, float]:
        pass

    def get_x(self) -> dict:
        return self.x

    def get_y(self) -> dict:
        return self.y

    def get_u(self) -> dict:
        return self.u

    def get_z(self) -> dict:
        return self.z

    def get_model(self) -> gp.Model | xp.problem:
        return self.model
