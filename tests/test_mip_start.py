import os

import gurobipy as gp

from trigger_arc_tsp.gurobi_model import GurobiModel
from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR


def test_get_variables_from_tour() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_2.txt"))

    tour = [0, 2, 1, 4, 3]
    x, y, u, z = inst.get_variables_from_tour(tour)

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
    assert z == {(*a, *b): u[a[0]] + 1 <= u[b[0]] for a in inst.edges for b in inst.edges if a != b}


def test_get_mip_start_from_saved_solution() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_2.txt"))

    # we know that we saved the solution [0, 3, 2, 1, 4] for this instance
    x, y, u, _ = inst.get_mip_start()

    assert sum(v for v in y.values()) <= 2
    assert sum(v for v in x.values()) == 5

    tour = sorted(u.keys(), key=lambda i: u[i])
    assert tour == [0, 3, 2, 1, 4]


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

    vars_ = inst.get_mip_start(use_tsp_only=True)
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


def test_that_improving_after_mip_start_improves_original_objective() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "instances_release_1/grf5.txt"))

    model_1 = GurobiModel(inst)
    model_1.formulate()

    tour_1 = [0, 13, 14, 9, 5, 15, 7, 18, 10, 6, 11, 17, 3, 12, 4, 2, 19, 8, 16, 1]
    obj_1 = inst.compute_objective(tour_1)
    assert obj_1 == 119.48

    # Get the optimization objective
    vars_tour_1 = inst.get_variables_from_tour(tour_1)
    model_1.provide_mip_start(vars_tour_1)
    gb_model_1 = model_1.get_model()
    gb_model_1.setParam(gp.GRB.Param.SolutionLimit, 1)
    gb_model_1.optimize()
    comp_tour_1, comp_cost_1 = model_1.get_original_solution()
    opt_objective_1 = gb_model_1.ObjVal

    # Check that the gurobi solution and objective is the same as the original
    assert round(comp_cost_1, 4) == 119.48
    assert comp_tour_1 == tour_1
    assert round(opt_objective_1, 4) == round(model_1.get_original_solution(keep_offset=True)[1], 4)

    model_2 = GurobiModel(inst)
    model_2.formulate()

    tour_2 = [0, 5, 15, 7, 12, 13, 4, 2, 8, 16, 1, 14, 9, 18, 10, 6, 11, 17, 3, 19]
    obj_2 = inst.compute_objective(tour_2)
    assert round(obj_2, 4) == 120.66

    # Get the optimization objective
    vars_tour_2 = inst.get_variables_from_tour(tour_2)
    model_2.provide_mip_start(vars_tour_2)
    gb_model_2 = model_2.get_model()
    gb_model_2.setParam(gp.GRB.Param.SolutionLimit, 1)
    gb_model_2.optimize()
    comp_tour_2, comp_cost_2 = model_2.get_original_solution()
    opt_objective_2 = gb_model_2.ObjVal

    # Check that the gurobi solution and objective is the same as the original
    assert round(comp_cost_2, 4) == 120.66
    assert comp_tour_2 == tour_2
    assert round(opt_objective_2, 4) == round(model_2.get_original_solution(keep_offset=True)[1], 4)

    # Since obj_1 < obj_2, we expect opt_objective_1 < opt_objective_2
    assert obj_1 < obj_2
    assert opt_objective_1 < opt_objective_2
