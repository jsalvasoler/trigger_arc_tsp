import os
from typing import Any, Generator

import gurobipy as gp
import pytest

from trigger_arc_tsp.gurobi_model import GurobiModel
from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR, MODELS_DIR


@pytest.fixture
def inst() -> Generator[Any, Any, Any]:
    if os.path.exists(os.path.join(MODELS_DIR, "test.mps")):
        os.remove(os.path.join(MODELS_DIR, "test.mps"))
    N = 3
    yield Instance(N=N, edges={e: 1 for e in [(0, 1), (1, 2), (2, 0)]}, relations={}, name="test.txt")
    os.remove(os.path.join(MODELS_DIR, "test.mps"))


def test_model_is_writen_after_formulating(inst: Instance) -> None:
    model = GurobiModel(inst)
    model.formulate()

    model_path = os.path.join(MODELS_DIR, inst.model_name)
    assert os.path.exists(model_path)
    os.remove(model_path)


def test_model_is_read_successfully(inst: Instance) -> None:
    model = GurobiModel(inst)
    model.formulate()  # Model is saved here

    model_path = os.path.join(MODELS_DIR, inst.model_name)

    model_2 = GurobiModel(inst)
    model_2.formulate(read_model=True)

    assert len(model.get_model().getVars()) == len(model_2.get_model().getVars())
    assert len(model.get_model().getConstrs()) == len(model_2.get_model().getConstrs())

    # Solve the model
    model_2.solve_model_with_parameters()

    assert model_2.get_model().Status == gp.GRB.OPTIMAL
    assert model_2.get_model().objVal == 3.0

    os.remove(model_path)


def test_variables_correctly_defined_when_reading_model(inst: Instance) -> None:
    model = GurobiModel(inst)
    model.formulate()  # Model is saved here

    del model

    model = GurobiModel(inst)
    model.formulate(read_model=True)  # Model is read here

    x, y, u, z = model.get_x(), model.get_y(), model.get_u(), model.get_z()
    assert len(x) == inst.A
    assert all(e in x for e in inst.edges)
    assert all(x[e].vType == gp.GRB.BINARY for e in x)
    assert len(y) == 0
    assert len(u) == inst.N
    assert all(i in u for i in inst.nodes)
    assert all(u[i].vType == gp.GRB.CONTINUOUS for i in u if i != 0)
    assert u[0] == 0
    assert len(z) == len(inst.z_var_indices)
    assert all(e in z for e in model.z_var_indices)
    assert all(z[e].vType == gp.GRB.BINARY for e in z if e[0] != 0)
    assert all(isinstance(z[e], int) for e in z if e[0] == 0)

    model.solve_model_with_parameters(mip_start=True)

    assert model.get_model().Status == gp.GRB.OPTIMAL
    assert model.get_model().objVal == 3.0


def test_starting_solution_for_saved_model() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "instances_release_1/grf5.txt"))
    model = GurobiModel(inst)
    model.formulate()  # Model is saved here

    del model
    model = GurobiModel(inst)
    model.formulate(read_model=True)  # Model is read here

    x, y, u, z = model.get_x(), model.get_y(), model.get_u(), model.get_z()
    assert len(x) == inst.A
    assert all(e in x for e in inst.edges)
    assert all(x[e].vType == gp.GRB.BINARY for e in x)
    assert len(y) == inst.R
    assert all(e in y for e in inst.relations)
    assert all(y[e].vType == gp.GRB.BINARY for e in y)
    assert len(u) == inst.N
    assert all(i in u for i in inst.nodes)
    assert all(u[i].vType == gp.GRB.CONTINUOUS for i in u if i != 0)
    assert u[0] == 0
    assert len(z) == len(inst.z_var_indices)
    assert all(e in z for e in model.z_var_indices)
    assert all(z[e].vType == gp.GRB.BINARY for e in z if e[0] != 0)
    assert all(isinstance(z[e], int) for e in z if e[0] == 0)

    starting_tour = [0, 16, 13, 4, 12, 17, 14, 9, 19, 6, 11, 3, 2, 8, 18, 10, 5, 15, 7, 1]
    vars_ = inst.get_variables_from_tour(starting_tour)
    model.provide_mip_start(vars_)
    gu_model = model.get_model()
    gu_model.setParam(gp.GRB.Param.SolutionLimit, 1)
    model.solve_model_with_parameters(mip_start=False)

    assert gu_model.NodeCount == 0
    assert round(gu_model.objVal, 4) == 119.94


def test_read_model_but_no_model_found() -> None:
    if os.path.exists(os.path.join(MODELS_DIR, "test.mps")):
        os.remove(os.path.join(MODELS_DIR, "test.mps"))

    inst = Instance(N=3, edges={(0, 1): 1, (1, 2): 1, (2, 0): 1}, relations={}, name="test.txt")
    model = GurobiModel(inst)
    model.formulate(read_model=True)
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL
    assert model.get_model().objVal == 3.0
