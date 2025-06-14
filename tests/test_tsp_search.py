import os
from copy import copy

import numpy as np

from trigger_arc_tsp.gurobi_tsp_model import GurobiTSPModel
from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.tsp_search import HeuristicSearch, TSPPrior, TSPPriorEval
from trigger_arc_tsp.utils import INSTANCES_DIR, fisher_yates_shuffle


def test_compute_node_distances_1() -> None:
    inst = Instance(
        N=3,
        edges={(0, 1): 1, (1, 2): 1, (2, 0): 1},
        relations={},
        name="test",
    )
    tsp_search = TSPPriorEval(inst)

    node_priorities = [0, 1, 2]
    node_dist = tsp_search.compute_node_dist(node_priorities)

    assert set(node_dist.values()) == {1}


def test_compute_node_distances_2() -> None:
    inst = Instance(
        N=5,
        edges={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 0): 1},
        relations={},
        name="test",
    )
    tsp_search = TSPPriorEval(inst)

    node_priorities = [0, 1, 2, 3, 4]
    node_dist = tsp_search.compute_node_dist(node_priorities)

    assert set(node_dist.values()) == {1, 2}
    assert node_dist[0, 4] == 1
    assert node_dist[1, 4] == 2


def test_tsp_search_solution_finds_only_feasible() -> None:
    inst = Instance(N=3, edges={(0, 1): 1, (1, 2): 1, (2, 0): 1}, relations={}, name="test")

    tsp_search = TSPPriorEval(inst)
    tsp_prior = TSPPrior(priorities=[0, 2, 1], alpha=0.5, beta=0.5)
    tour, cost = tsp_search.evaluate_individual(tsp_prior=tsp_prior)

    # There is only one feasible tour
    assert tour == [0, 1, 2]
    assert cost == 3


def test_get_edges_for_tsp_search_1() -> None:
    inst = Instance(
        N=3,
        edges={(0, 1): 1, (1, 2): 1, (2, 0): 1},
        relations={(0, 1, 2, 0): 0},
        name="test",
    )

    tsp_search = TSPPriorEval(inst)
    tsp_prior = TSPPrior(priorities=[0, 1, 2], alpha=0.5, beta=0.5)
    edges = tsp_search.get_edges_for_tsp_search(tsp_prior=tsp_prior)
    assert edges[0, 1] < edges[1, 2]
    assert edges[2, 0] < edges[1, 2]


def test_get_edges_for_tsp_search_2() -> None:
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 2), (2, 1), (4, 0)]
    inst = Instance(
        N=5,
        edges=dict.fromkeys(edges, 1),
        relations={(0, 1, 3, 4): -1, (0, 2, 3, 4): -1},
        name="test",
    )
    tsp_search = TSPPriorEval(inst)
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
    relation_active_prob = {
        r: edge_used_prob[r[0], r[1]] * edge_used_prob[r[2], r[3]] for r in inst.relations
    }
    assert relation_active_prob[0, 1, 3, 4] > relation_active_prob[0, 2, 3, 4]

    # Do the same for tour_2
    edge_used_prob = {edge: 1 / dist_2[edge] for edge in inst.edges}
    relation_active_prob = {
        r: edge_used_prob[r[0], r[1]] * edge_used_prob[r[2], r[3]] for r in inst.relations
    }
    assert relation_active_prob[0, 1, 3, 4] < relation_active_prob[0, 2, 3, 4]

    tsp_prior_1 = TSPPrior(priorities=tour_1, alpha=0.5, beta=0.5)
    edges_1 = tsp_search.get_edges_for_tsp_search(tsp_prior_1)
    assert edges_1[0, 1] < edges_1[0, 2]
    tsp_prior_2 = TSPPrior(priorities=tour_2, alpha=0.5, beta=0.5)
    edges_2 = tsp_search.get_edges_for_tsp_search(tsp_prior_2)
    assert edges_2[0, 2] < edges_2[0, 1]

    assert edges_1 != edges_2

    # The two paths are symmetrical on the instance graph
    assert set(edges_1.values()) == set(edges_2.values())
    assert edges_1[0, 1] == edges_2[0, 2]
    assert edges_1[0, 2] == edges_2[0, 1]

    tsp_prior_1 = TSPPrior(priorities=tour_1, alpha=3, beta=0.5)
    tsp_prior_2 = TSPPrior(priorities=tour_2, alpha=3, beta=0.5)
    edges_1 = tsp_search.get_edges_for_tsp_search(tsp_prior_1)
    edges_2 = tsp_search.get_edges_for_tsp_search(tsp_prior_2)
    assert set(edges_1.values()) == set(edges_2.values())
    assert edges_1[0, 1] == edges_2[0, 2]
    assert edges_1[0, 2] == edges_2[0, 1]


def test_tsp_search_2() -> None:
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 2), (2, 1), (4, 0)]
    inst = Instance(
        N=5,
        edges=dict.fromkeys(edges, 1),
        relations={(0, 1, 3, 4): -1, (0, 2, 3, 4): -1},
        name="test",
    )
    tsp_search = TSPPriorEval(inst)

    tour_1 = [0, 1, 2, 3, 4]
    tour_2 = [0, 2, 1, 3, 4]
    tsp_prior_1 = TSPPrior(priorities=tour_1, alpha=1, beta=0.5)
    tsp_prior_2 = TSPPrior(priorities=tour_2, alpha=1, beta=0.5)
    edges_1 = tsp_search.get_edges_for_tsp_search(tsp_prior_1)
    edges_2 = tsp_search.get_edges_for_tsp_search(tsp_prior_2)
    assert edges_1[0, 1] == edges_2[0, 2]
    assert edges_2[0, 1] == edges_1[0, 2]

    comp_tour_1, _ = tsp_search.evaluate_individual(tsp_prior_1)
    comp_tour_2, _ = tsp_search.evaluate_individual(tsp_prior_2)

    assert comp_tour_1 == tour_1
    assert comp_tour_2 == tour_2


def test_tsp_search_3() -> None:
    inst = Instance.load_instance_from_file(
        os.path.join(INSTANCES_DIR, "examples/example_2.txt"),
    )
    tsp_search = TSPPriorEval(inst)
    dist = tsp_search.compute_node_dist([0, 2, 1, 4, 3])
    assert all(d > 0 for d in dist.values())

    # The two feasible tours of this instance graph. Worse obj of the two is 72.0
    tour_1 = [0, 2, 1, 4, 3]
    tour_2 = [0, 3, 2, 1, 4]
    tour, cost = tsp_search.evaluate_individual(TSPPrior(priorities=tour_1, alpha=1, beta=0.5))
    assert tour in [tour_1, tour_2]
    assert cost <= 72.0

    tour, cost = tsp_search.evaluate_individual(TSPPrior(priorities=tour_2, alpha=1, beta=0.5))
    assert tour in [tour_1, tour_2]
    assert cost <= 72.0


def test_generate_random_permutation() -> None:
    test_list = [1, 2, 3, 4, 5]
    number_of_times_i_in_j = {(i, j): 0 for i in test_list for j in range(len(test_list))}
    for _ in range(1000):
        perm = fisher_yates_shuffle(test_list)
        for j, i in enumerate(perm):
            number_of_times_i_in_j[i, j] += 1

    avg_pos_j = {
        j: np.mean([number_of_times_i_in_j[i, j] for i in test_list])
        for j in range(len(test_list))
    }
    assert len(perm) == len(test_list) == len(set(perm))
    assert all(abs(avg_pos_j[j] - 200) < 150 for j in range(len(test_list)))


def test_best_among_multiple_tsp_feasible_solutions_is_selected() -> None:
    inst = Instance.load_instance_from_file(
        os.path.join(INSTANCES_DIR, "instances_release_1/grf4.txt"),
    )

    tsp_prior = TSPPrior(priorities=list(range(inst.N)), alpha=0, beta=0.5)
    tsp_search = TSPPriorEval(inst)
    edges = tsp_search.get_edges_for_tsp_search(tsp_prior)
    tsp_inst = Instance(N=inst.N, edges=edges, relations={}, name="grf4_to_tsp")
    tsp_model = GurobiTSPModel(tsp_inst)
    tsp_model.formulate()
    tsp_model.solve_to_optimality()
    best_5_tours = tsp_model.get_best_n_tours(5)

    assert len(best_5_tours) == 5
    best_5_tours = sorted(best_5_tours, key=lambda t: inst.compute_objective(t))
    best_tour = best_5_tours[0]
    best_cost = inst.compute_objective(best_tour)

    best_tour_search, best_cost_search = tsp_search.evaluate_individual(tsp_prior=tsp_prior)
    assert best_cost_search == best_cost
    assert best_tour_search == best_tour


def test_run_post_trials_just_one_trial() -> None:
    inst = Instance.load_instance_from_file(
        os.path.join(INSTANCES_DIR, "examples/example_2.txt"),
    )
    priors_to_test = [
        TSPPrior(priorities=[0, 2, 1, 4, 3], alpha=1, beta=0.5, cost=0, rel_gap=0.5),
        TSPPrior(priorities=[0, 3, 2, 1, 4], alpha=1, beta=0.5, cost=1, rel_gap=0.5),
    ]

    prior_search = HeuristicSearch(inst, search_type="randomized")
    best_tour, best_cost = prior_search.run_post_trials(
        priors_to_test, best_cost=float("inf"), best_tour=list(range(inst.N)), n_post_trials=1
    )
    # post-trials will have evaluated the first one because it has cost 0
    tour, cost = prior_search.searcher.evaluate_individual(priors_to_test[0])
    assert best_cost == cost
    assert best_tour == tour


def test_run_post_trials_selects_best_one() -> None:
    inst = Instance.load_instance_from_file(
        os.path.join(INSTANCES_DIR, "instances_release_1/grf4.txt"),
    )
    tour_1 = [0, 13, 17, 12, 6, 4, 14, 11, 5, 1, 2, 19, 18, 15, 9, 10, 16, 8, 7, 3]
    tour_2 = [0, 8, 17, 12, 6, 4, 14, 11, 5, 1, 19, 18, 2, 9, 15, 10, 16, 13, 7, 3]
    priors_to_test = [
        TSPPrior(priorities=tour_1, alpha=0, beta=0.5, cost=0, rel_gap=0.5),
        TSPPrior(priorities=tour_2, alpha=0, beta=0.5, cost=1, rel_gap=0.5),
    ]
    prior_search = HeuristicSearch(inst, search_type="randomized")
    best_cost = min(
        prior_search.searcher.evaluate_individual(copy(prior))[1] for prior in priors_to_test
    )

    _, best_cost_comp = prior_search.run_post_trials(
        priors_to_test, best_cost=float("inf"), best_tour=tour_1, n_post_trials=2
    )
    assert best_cost_comp == best_cost


def test_swap_2_search_tour_generatior() -> None:
    prior_search = HeuristicSearch(
        Instance(
            N=5,
            edges={(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 0): 1},
            relations={},
            name="test",
        ),
        search_type="swap_2",
    )
    assert len(list(prior_search.generate_swap_2_permutations([0, 1, 2, 3, 4]))) == 4 * 3 / 2


def test_swap_2_search_runs() -> None:
    inst = Instance.load_instance_from_file(
        os.path.join(INSTANCES_DIR, "examples/example_2.txt")
    )
    prior_search = HeuristicSearch(inst, search_type="swap_2")
    prior_search.run(n_trials=None, n_post_trials=1)

    assert True


def test_delay_1_search_generation() -> None:
    inst = Instance.load_instance_from_file(
        os.path.join(INSTANCES_DIR, "examples/example_2.txt"),
    )
    prior_search = HeuristicSearch(inst, search_type="delay_2")
    priors = prior_search.generate_delay_1_node_permutations([0, 1, 2, 3, 4])
    for _ in priors:
        pass

    assert True
