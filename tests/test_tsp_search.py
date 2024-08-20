import os

from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.tsp_search import TSPSearch
from trigger_arc_tsp.utils import INSTANCES_DIR


def test_compute_node_distances_1() -> None:
    inst = Instance(N=3, edges={(0, 1): 1, (1, 2): 1, (2, 0): 1}, relations={}, name="test")
    tsp_search = TSPSearch(inst)

    node_priorities = [0, 1, 2]
    node_dist = tsp_search.compute_node_dist(node_priorities)

    assert set(node_dist.values()) == {1}


def test_compute_node_distances_2() -> None:
    inst = Instance(N=5, edges={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 0): 1}, relations={}, name="test")
    tsp_search = TSPSearch(inst)

    node_priorities = [0, 1, 2, 3, 4]
    node_dist = tsp_search.compute_node_dist(node_priorities)

    assert set(node_dist.values()) == {1, 2}
    assert node_dist[0, 4] == 1
    assert node_dist[1, 4] == 2


def test_tsp_search_solution_finds_only_feasible() -> None:
    inst = Instance(N=3, edges={(0, 1): 1, (1, 2): 1, (2, 0): 1}, relations={}, name="test")

    tsp_search = TSPSearch(inst)
    tour, cost = tsp_search.evaluate_individual([2, 1, 0], 0.5)

    # There is only one feasible tour
    assert tour == [0, 1, 2]
    assert cost == 3


def test_get_edges_for_tsp_search_1() -> None:
    inst = Instance(N=3, edges={(0, 1): 1, (1, 2): 1, (2, 0): 1}, relations={(0, 1, 2, 0): 0}, name="test")

    tsp_search = TSPSearch(inst)
    edges = tsp_search.get_edges_for_tsp_search([0, 1, 2], alpha=0.5)
    assert edges[0, 1] < edges[1, 2]
    assert edges[2, 0] < edges[1, 2]


def test_get_edges_for_tsp_search_2() -> None:
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 2), (2, 1), (4, 0)]
    inst = Instance(N=5, edges={e: 1 for e in edges}, relations={(0, 1, 3, 4): -1, (0, 2, 3, 4): -1}, name="test")
    tsp_search = TSPSearch(inst)
    tour_1 = [0, 1, 2, 3, 4]
    tour_2 = [0, 2, 1, 3, 4]
    dist_1 = tsp_search.compute_node_dist(tour_1)
    dist_2 = tsp_search.compute_node_dist(tour_2)
    assert set(dist_1.values()) == set(dist_2.values())
    assert dist_1[0, 1] == dist_1[1, 2] == dist_1[2, 3] == dist_1[3, 4]
    assert dist_2[0, 2] == dist_2[2, 1] == dist_2[1, 3] == dist_2[3, 4]
    assert dist_1[1, 3] == dist_2[2, 3] == 2

    # Manually compute the probability of relations being active in tour_1
    edge_used_prob = {edge: 1 / dist_1[edge] for edge in inst.edges}
    relation_active_prob = {r: edge_used_prob[r[0], r[1]] * edge_used_prob[r[2], r[3]] for r in inst.relations}
    assert relation_active_prob[0, 1, 3, 4] > relation_active_prob[0, 2, 3, 4]

    # Do the same for tour_2
    edge_used_prob = {edge: 1 / dist_2[edge] for edge in inst.edges}
    relation_active_prob = {r: edge_used_prob[r[0], r[1]] * edge_used_prob[r[2], r[3]] for r in inst.relations}
    assert relation_active_prob[0, 1, 3, 4] < relation_active_prob[0, 2, 3, 4]

    edges_1 = tsp_search.get_edges_for_tsp_search(tour_1, alpha=0.5)
    assert edges_1[0, 1] < edges_1[0, 2]
    edges_2 = tsp_search.get_edges_for_tsp_search(tour_2, alpha=0.5)
    assert edges_2[0, 2] < edges_2[0, 1]

    assert edges_1 != edges_2

    # The two paths are symmetrical on the instance graph
    assert set(edges_1.values()) == set(edges_2.values())
    assert edges_1[0, 1] == edges_2[0, 2]
    assert edges_1[0, 2] == edges_2[0, 1]

    edges_1 = tsp_search.get_edges_for_tsp_search(tour_1, alpha=3)
    edges_2 = tsp_search.get_edges_for_tsp_search(tour_2, alpha=3)
    assert set(edges_1.values()) == set(edges_2.values())
    assert edges_1[0, 1] == edges_2[0, 2]
    assert edges_1[0, 2] == edges_2[0, 1]


def test_tsp_search_2() -> None:
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 2), (2, 1), (4, 0)]
    inst = Instance(N=5, edges={e: 1 for e in edges}, relations={(0, 1, 3, 4): -1, (0, 2, 3, 4): -1}, name="test")
    tsp_search = TSPSearch(inst)

    tour_1 = [0, 1, 2, 3, 4]
    tour_2 = [0, 2, 1, 3, 4]
    edges_1 = tsp_search.get_edges_for_tsp_search(tour_1, alpha=1)
    edges_2 = tsp_search.get_edges_for_tsp_search(tour_2, alpha=1)
    assert edges_1[0, 1] == edges_2[0, 2]
    assert edges_2[0, 1] == edges_1[0, 2]

    comp_tour_1, _ = tsp_search.evaluate_individual(tour_1, alpha=1)
    comp_tour_2, _ = tsp_search.evaluate_individual(tour_2, alpha=1)

    assert comp_tour_1 == tour_1
    assert comp_tour_2 == tour_2


def test_tsp_search_3() -> None:
    inst = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, "examples/example_2.txt"))
    tsp_search = TSPSearch(inst)
    dist = tsp_search.compute_node_dist([0, 2, 1, 4, 3])
    assert all(d > 0 for d in dist.values())

    # The two feasible tours of this instance graph. Worse obj of the two is 72.0
    tour_1 = [0, 2, 1, 4, 3]
    tour_2 = [0, 3, 2, 1, 4]
    tour, cost = tsp_search.evaluate_individual(node_priorities=tour_1)
    assert tour in [tour_1, tour_2]
    assert cost <= 72.0

    tour, cost = tsp_search.evaluate_individual(node_priorities=tour_2)
    assert tour in [tour_1, tour_2]
    assert cost <= 72.0
