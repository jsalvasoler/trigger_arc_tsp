import os

import gurobipy as gp

from trigger_arc_tsp.gurobi_model import GurobiModel
from trigger_arc_tsp.gurobi_tsp_model import GurobiTSPModel
from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR


def test_tsp_model_1() -> None:
    isnt = Instance(N=3, edges={(0, 1): 1, (1, 2): 1, (2, 0): 1}, relations={}, name="test")
    model = GurobiTSPModel(isnt)
    model.formulate()
    model.solve_to_feasible_solution()

    tour = model.get_tour()

    assert tour == [0, 1, 2]


def test_tsp_model_2() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_1.txt"))
    model = GurobiTSPModel(inst)
    model.formulate()
    model.solve_to_feasible_solution()

    tour = model.get_tour()
    # Check that the path is feasible
    edges_used = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)] + [(tour[-1], tour[0])]
    assert all(e in inst.edges for e in edges_used)


def test_get_variables_from_tour() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_2.txt"))

    tour = [0, 2, 1, 4, 3]
    x, y, u, z_1, z_2 = inst.get_variables_from_tour(tour)

    edges_used = {(tour[i], tour[i + 1] if i + 1 < len(tour) else tour[0]) for i in range(len(tour))}

    # check x
    assert all(x[e] == 1 for e in edges_used)
    assert all(x[e] == 0 for e in set(inst.edges) - edges_used)

    # check u
    for edge in edges_used:
        if edge[1] == 0:
            continue
        assert u[edge[0]] < u[edge[1]]

    # check y
    for rel in inst.relations:
        if rel == (2, 1, 1, 4):
            assert y[rel] == 1
        else:
            assert y[rel] == 0

    # check z_1 and z_2
    assert z_1 == {(a, b, c): u[c[0]] <= u[b[0]] for a, b, c in z_1}
    assert z_2 == {(a, b, c): u[a[0]] <= u[c[0]] for a, b, c in z_2}


def test_get_mip_start() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_2.txt"))

    x, y, _, _, _ = inst.get_mip_start()

    assert sum(v for v in y.values()) <= 2
    assert sum(v for v in x.values()) == 5


def test_provide_mip_start_1() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_2.txt"))

    model = GurobiModel(inst)
    model.formulate()

    vars_ = inst.get_mip_start()
    model.provide_mip_start(vars_)

    gb_model = model.get_model()
    gb_model.setParam(gp.GRB.Param.SolutionLimit, 1)

    gb_model.optimize()

    assert gb_model.NodeCount == 0


def test_provide_mip_start_2() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "instances_release_1/grf5.txt"))

    model = GurobiModel(inst)
    model.formulate()

    vars_ = inst.get_mip_start()
    model.provide_mip_start(vars_)

    gb_model = model.get_model()
    gb_model.setParam(gp.GRB.Param.SolutionLimit, 1)

    gb_model.optimize()

    assert gb_model.NodeCount == 0


def test_provide_mip_start_3() -> None:
    edges = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}
    relations = {}
    inst = Instance(N=4, edges=edges, relations=relations, name="test")

    model = GurobiModel(inst)
    model.formulate()

    vars_ = inst.get_mip_start()

    model.provide_mip_start(vars_)

    gb_model = model.get_model()
    gb_model.setParam(gp.GRB.Param.SolutionLimit, 1)

    gb_model.optimize()

    assert gb_model.NodeCount == 0
