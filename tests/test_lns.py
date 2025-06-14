from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.milp_lns import MILPbasedLNS


def test_lns_1() -> None:
    edges = [(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3), (3, 0)]
    edges_with_cost = dict.fromkeys(edges, 1)
    # The optimal tour should avoid (0,1) -> [0, 2, 1, 3]
    edges_with_cost[(0, 1)] = 10
    inst = Instance(4, edges_with_cost, relations={}, name="test")

    current_tour = [0, 1, 2, 3]
    reinsert_nodes = [1, 2]

    lns_model = MILPbasedLNS(inst, current_tour, reinsert_nodes)
    new_tour, cost = lns_model.explore()

    assert cost == 4
    assert new_tour == [0, 2, 1, 3]


def test_lns_reinserting_everything() -> None:
    edges = [(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3), (3, 0)]
    edges_with_cost = dict.fromkeys(edges, 1)
    # The optimal tour should avoid (0,1) -> [0, 2, 1, 3]
    edges_with_cost[(0, 1)] = 10
    inst = Instance(4, edges_with_cost, relations={}, name="test")

    current_tour = [0, 1, 2, 3]
    reinsert_nodes = [0, 1, 2, 3]

    lns_model = MILPbasedLNS(inst, current_tour, reinsert_nodes)
    new_tour, cost = lns_model.explore()

    assert cost == 4
    assert new_tour == [0, 2, 1, 3]


def test_lns_2() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_2.txt")

    current_tour = [0, 2, 1, 4, 3]
    current_cost = 71
    assert inst.compute_objective(current_tour) == current_cost
    reinsert_nodes = [0, 3]

    lns_model = MILPbasedLNS(inst, current_tour, reinsert_nodes)
    new_tour, new_cost = lns_model.explore()

    # Known optimal tour
    best_tour = [0, 3, 2, 1, 4]
    best_cost = 62
    assert inst.compute_objective(best_tour) == best_cost

    assert new_cost == best_cost
    assert new_tour == best_tour


def test_lns_reinserting_nothing() -> None:
    inst = Instance.load_instance_from_file("instances/examples/example_2.txt")

    current_tour = [0, 2, 1, 4, 3]
    current_cost = 71
    assert inst.compute_objective(current_tour) == current_cost

    lns_model = MILPbasedLNS(inst, current_tour, reinsert_nodes=[])
    new_tour, new_cost = lns_model.explore()
    assert current_tour == new_tour
    assert current_cost == new_cost
