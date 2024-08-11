import gurobipy as gp
import pytest

from trigger_arc_tsp.gurobi_model import GurobiModel
from trigger_arc_tsp.instance import Instance


def test_solving_non_formulated_model() -> None:
    N = 3
    inst = Instance(N=N, edges={e: 1 for e in [(0, 1), (1, 2), (2, 0)]}, relations={}, name="test")
    model = GurobiModel(inst)

    with pytest.raises(ValueError, match="Model is not formulated"):
        model.solve_model_with_parameters()

    model.formulate()
    model.solve_model_with_parameters()


def test_gurobi_model_sample_1() -> None:
    N = 3
    inst = Instance(N=N, edges={e: 1 for e in [(0, 1), (1, 2), (2, 0)]}, relations={}, name="test")
    model = GurobiModel(inst)
    model.formulate()
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL
    assert model.get_model().objVal == 3.0

    x = model.get_x()
    assert x[0, 1].x == 1.0
    assert x[1, 2].x == 1.0
    assert x[2, 0].x == 1.0

    assert inst.compute_objective([0, 1, 2, 0]) == 3.0


def test_gurobi_model_sample_2() -> None:
    N = 4
    inst = Instance(
        N=N, edges={e: 1 for e in [(0, 1), (1, 2), (2, 3), (3, 0)]}, relations={(0, 1, 1, 2): 2}, name="test"
    )
    model = GurobiModel(inst)
    model.formulate()
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL

    x = model.get_x()
    assert x[0, 1].x == 1.0
    assert x[1, 2].x == 1.0
    assert x[2, 3].x == 1.0
    assert x[3, 0].x == 1.0

    y = model.get_y()
    assert y[0, 1, 1, 2].x == 1.0

    tour, cost = model.get_original_objective()
    assert cost == 5.0
    assert tour == [0, 1, 2, 3]

    assert inst.compute_objective([0, 1, 2, 3, 0]) == 5.0


def test_gurobi_model_sample_3() -> None:
    N = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges = {e: 1 for e in edges}
    relations = {(0, 1, 3, 0): 5, (1, 2, 3, 0): 10}

    inst = Instance(N=N, edges=edges, relations=relations, name="test")
    model = GurobiModel(inst)
    model.formulate()
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL
    tour, cost = model.get_original_objective()

    # The only feasible tour has the active edge (1, 2)
    assert tour == [0, 1, 2, 3]
    assert cost == 1 + 1 + 1 + 10

    assert inst.compute_objective([0, 1, 2, 3, 0]) == 13.0


def test_gurobi_model_sample_4() -> None:
    N = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges = {e: 1 for e in edges}
    relations = {(0, 1, 2, 3): 2, (1, 2, 2, 3): 10, (3, 0, 2, 3): 2}

    inst = Instance(N=N, edges=edges, relations=relations, name="test")
    model = GurobiModel(inst)
    model.formulate()
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL
    tour, cost = model.get_original_objective()

    # The only feasible tour has the active edge (1, 2)
    assert tour == [0, 1, 2, 3]
    assert cost == 1 + 1 + 10 + 1

    assert inst.compute_objective([0, 1, 2, 3, 0]) == 13.0


def test_gurobi_model_sample_5() -> None:
    N = 2
    edges = [(0, 1), (1, 0)]
    edges = {e: 1 for e in edges}
    relations = {(0, 1, 1, 0): 2}

    inst = Instance(N=N, edges=edges, relations=relations, name="test")
    model = GurobiModel(inst)
    model.formulate()
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL
    tour, cost = model.get_original_objective()

    # The only feasible tour has the active edge (0, 1)
    assert tour == [0, 1]
    assert cost == 1 + 2

    assert inst.compute_objective([0, 1, 0]) == 3.0


def test_gurobi_model_sample_6() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_2.txt")
    model = GurobiModel(inst)
    model.formulate()
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL
    tour, cost = model.get_original_objective()

    assert inst.compute_objective(tour) == cost

    assert cost == 62.0
    assert tour == [0, 3, 2, 1, 4]

    assert inst.compute_objective([0, 3, 2, 1, 4, 0]) == 62.0


def test_gurobi_model_sample_7() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_3.txt")
    model = GurobiModel(inst)
    model.formulate()
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL
    tour, cost = model.get_original_objective()

    assert inst.compute_objective(tour) == cost

    assert cost == 40.0
    assert tour == [0, 1, 2, 3, 4]

    assert inst.compute_objective([0, 1, 2, 3, 4]) == 40.0
    assert inst.compute_objective([0, 4, 3, 2, 1]) == 50.0


def test_gurobi_model_sample_8() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_4.txt")
    model = GurobiModel(inst)
    model.formulate()
    model.solve_model_with_parameters()

    assert model.get_model().Status == gp.GRB.OPTIMAL
    tour, cost = model.get_original_objective()

    assert inst.compute_objective(tour) == cost

    assert cost == 50.0
    assert tour == [0, 4, 3, 2, 1]

    assert inst.compute_objective([0, 4, 3, 2, 1]) == 50.0
    assert inst.compute_objective([0, 1, 2, 3, 4]) == 60.0
