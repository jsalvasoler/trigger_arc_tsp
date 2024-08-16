import os

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


def test_tsp_big_model() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "instances_release_1/grf1.txt"))
    model = GurobiTSPModel(inst)
    model.formulate()

    model.solve_to_optimality()
    tour = model.get_tour()
    print(tour)
    assert len(tour) == inst.N
