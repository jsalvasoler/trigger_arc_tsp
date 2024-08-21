from __future__ import annotations

import os
import random

from trigger_arc_tsp.gurobi_tsp_model import GurobiTSPModel
from trigger_arc_tsp.instance import Instance
from trigger_arc_tsp.utils import INSTANCES_DIR, cleanup_instance_name, fisher_yates_shuffle


class TSPSearcher:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    def evaluate_individual(self, node_priorities: list, alpha: float = 1, beta: float = 0.5) -> list[list, float]:
        assert len(node_priorities) == self.instance.N

        tsp_edges = self.get_edges_for_tsp_search(node_priorities, alpha=alpha, beta=beta)
        tsp_instance = Instance(
            N=self.instance.N,
            edges=tsp_edges,
            relations={},
            name=self.instance.name,
        )
        self.model = GurobiTSPModel(tsp_instance)
        self.model.formulate()
        self.model.solve_to_optimality(logs=False, time_limit_sec=10)
        tour = self.model.get_best_tour()
        cost = self.instance.compute_objective(tour)
        return tour, cost

    def compute_node_dist(self, node_priorities: list) -> dict:
        node_dist = {}
        for i, node_i in enumerate(node_priorities):
            for j, node_j in enumerate(node_priorities):
                if i == j:
                    node_dist[node_i, node_j] = 1
                else:
                    node_dist[node_i, node_j] = min(abs(i - j), self.instance.N - abs(i - j))
        return node_dist

    def get_edges_for_tsp_search(self, node_priorities: list, alpha: float = 1, beta: float = 0.5) -> dict:
        # 1. Estimate the probability that an edge is used.
        # This is done with a probability inversely proportional to the
        # distance between the nodes in the node_priorities list

        node_dist = self.compute_node_dist(node_priorities)

        edge_used_prob = {edge: 1 / node_dist[edge] for edge in self.instance.edges}

        # 2. Estimate the probability that a relation is active.
        # This is done considering (a) the probability that the trigger is used,
        # (b) the probability that the target is used, and (c) the closeness
        # between the trigger and the target
        relation_active_prob = {
            r: edge_used_prob[r[0], r[1]] * edge_used_prob[r[2], r[3]] * (1 / (node_dist[r[1], r[2]] ** beta))
            for r in self.instance.relations
        }

        edges_cost = self.instance.edges.copy()
        for a in self.instance.R_a:
            for b in self.instance.R_a[a]:
                val = alpha * relation_active_prob[*b, *a] * self.instance.relations[*b, *a]
                edges_cost[a] += val
                edges_cost[b] += val
        return edges_cost


class PriorRandomizedSearch:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.searcher = TSPSearcher(self.instance)

    def generate_random_permutation(self) -> list:
        return [0, *fisher_yates_shuffle(list(range(1, self.instance.N)))]

    def run(self, n_trials: int = 60) -> None:
        # Generate n_trials random permutations of 0, ..., self.instance.N-1
        # Evaluate all of them

        best_tour = None
        best_cost = float("inf")

        for i in range(n_trials):
            node_priorities = self.generate_random_permutation()
            tour, cost = self.searcher.evaluate_individual(
                node_priorities, alpha=random.uniform(0.1, 3), beta=random.uniform(0.1, 3)
            )
            improved = cost < best_cost
            if improved:
                best_tour = tour
                best_cost = cost
            self.print_log_line(i, cost, best_cost, improved=improved)

        print(f"Best tour: {best_tour} - Cost: {best_cost}")
        assert best_cost == self.instance.compute_objective(best_tour)
        self.instance.save_solution(best_tour, best_cost)

    def print_log_line(self, it: int, cost: float, best_cost: float, *, improved: bool) -> None:
        # Define the width for each column
        it_width = 5  # Width for the iteration number
        cost_width = 15  # Width for the cost
        best_cost_width = 15  # Width for the best cost

        # Format the string with specified widths
        s = (
            f"Iteration {it:<{it_width}}"
            f"Cost: {cost:<{cost_width}.6f}"
            f"Best cost: {best_cost:<{best_cost_width}.6f}"
        )

        if improved:
            s += " (*)"

        print(s)


def prior_randomized_search(instance_name: str, n_trials: int) -> None:
    instance_name = cleanup_instance_name(instance_name)
    instance = Instance.load_instance_from_file(os.path.join(INSTANCES_DIR, instance_name))
    prior_randomized_search = PriorRandomizedSearch(instance)
    prior_randomized_search.run(n_trials=n_trials)


if __name__ == "__main__":
    prior_randomized_search(
        instance_name="instances_release_1/grf5.txt",
        n_trials=60,
    )
